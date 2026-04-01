"""KawaiiLLM inference engine — faithful replica of training forward pass.

Loads a KawaiiLLM checkpoint (MemE + Projector + LLM) and provides:
  - set_memory(): encode context through MemE -> Projector, cache prefix embeddings
  - generate(): build LLM input (prefix + conversation) and generate with streaming

The encode/assembly logic replicates model.py encode_context() (lines 271-367)
and forward() non-NTP path (lines 488-589) exactly for B=1 inference.
"""

import logging
import os
from threading import Event, Thread
from typing import Iterator, Optional, Union

import torch
from transformers import AutoTokenizer, TextIteratorStreamer

from src.train.model import KawaiiLLMModel, SPECIAL_TOKENS

logger = logging.getLogger(__name__)


class KawaiiInferenceEngine:
    """Load a KawaiiLLM checkpoint and generate text with memory context."""

    def __init__(
        self,
        checkpoint_dir: str,
        num_mem_tokens: int = 128,
        device: str = "cuda",
        attn_implementation: Optional[str] = None,
    ):
        self.device = torch.device(device)
        self.num_mem_tokens = num_mem_tokens

        # --- Tokenizer (same sequence as train.py:58-74) ---
        # Trainer saves tokenizer to checkpoint root, not llm/ subdir
        llm_dir = os.path.join(checkpoint_dir, "llm")
        tokenizer_dir = (
            llm_dir
            if os.path.isfile(os.path.join(llm_dir, "tokenizer_config.json"))
            else checkpoint_dir
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        num_added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": SPECIAL_TOKENS}
        )
        logger.info("Registered %d special tokens: %s", num_added, SPECIAL_TOKENS)

        # --- Model (same sequence as train.py:93-126) ---
        self.model = KawaiiLLMModel.from_checkpoint(
            checkpoint_dir=checkpoint_dir,
            num_mem_tokens=num_mem_tokens,
            freeze_meme=True,
            freeze_llm=True,
            freeze_projector=True,
            attn_implementation=attn_implementation,
        )
        self.model.meme.resize_token_embeddings(len(self.tokenizer))
        self.model.llm.resize_token_embeddings(len(self.tokenizer))

        special_token_ids = {
            tok: self.tokenizer.convert_tokens_to_ids(tok)
            for tok in SPECIAL_TOKENS
        }
        special_token_ids["pad_token_id"] = self.tokenizer.pad_token_id
        self.model.set_special_token_ids(special_token_ids)

        self.model.to(self.device).eval()
        logger.info("Model loaded on %s", self.device)

        # Cached memory state
        self._memory_prefix_embeds: Optional[torch.Tensor] = None  # [1, n_mem+2, 4096]
        self._memory_prefix_mask: Optional[torch.Tensor] = None    # [1, n_mem+2]
        self._active_n_mem: int = 0
        self._stop_event = Event()

    def stop(self):
        """Signal the current generation to stop."""
        self._stop_event.set()

    @torch.no_grad()
    def set_memory(self, memory_text: str, n_mem: Optional[int] = None) -> None:
        """Encode memory_text through MemE -> Projector, cache prefix embeddings.

        Replicates model.py encode_context() (lines 301-361) and forward()
        projection + prefix assembly (lines 496-537) exactly for B=1.
        """
        if not memory_text.strip():
            self._memory_prefix_embeds = None
            self._memory_prefix_mask = None
            self._active_n_mem = 0
            logger.info("Memory cleared")
            return

        n_mem = min(n_mem, self.num_mem_tokens) if n_mem is not None else self.num_mem_tokens
        self._active_n_mem = n_mem

        # Tokenize context (no left-padding needed for B=1)
        ctx_enc = self.tokenizer(
            memory_text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=4096,
        )
        context_ids = ctx_enc["input_ids"].to(self.device)              # [1, L]
        context_mask = ctx_enc["attention_mask"].to(self.device)        # [1, L]

        # --- Replicate encode_context (model.py:303-361) ---
        meme_embed = self.model.meme.get_input_embeddings()

        # Get text embeddings from MemE's embedding layer (model.py:306)
        text_embeds = meme_embed(context_ids)                           # [1, L, 2560]

        # Get MEM token (Query) embeddings (model.py:309-310)
        mem_embeds = self.model.mem_embeddings.weight[:n_mem]           # [n_mem, 2560]
        mem_embeds = mem_embeds.unsqueeze(0)                            # [1, n_mem, 2560]

        # Concatenate: [text, Q_1..Q_n] (model.py:314-316)
        combined = torch.cat([text_embeds, mem_embeds], dim=1)          # [1, L+n_mem, 2560]

        # Extend attention mask with ones for MEM tokens (model.py:319-326)
        extra_mask = torch.ones(1, n_mem, dtype=context_mask.dtype, device=self.device)
        extended_mask = torch.cat([context_mask, extra_mask], dim=1)    # [1, L+n_mem]

        # Fix position 0 to prevent NaN under causal mask (model.py:337)
        extended_mask[:, 0] = 1

        # Build position_ids from attention mask (model.py:340-341)
        position_ids = extended_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(extended_mask == 0, 0)

        # Run MemE + Projector under a single autocast scope (matches training
        # where the entire forward() runs under DeepSpeed/Accelerate mixed precision).
        # C1 fix: projection must also run in bfloat16, not float32.
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            outputs = self.model.meme(
                inputs_embeds=combined,
                attention_mask=extended_mask,
                position_ids=position_ids,
            )

            # Extract last n_mem positions (model.py:361)
            mem_hidden = outputs.last_hidden_state[:, -n_mem:, :]      # [1, n_mem, 2560]

            # --- Replicate projection (model.py:496-516) ---
            projected = self.model.projector(mem_hidden)               # [1, n_mem, 4096]

        # Add pad_embed residual base (model.py:511-516) — outside autocast
        # because llm_embed produces bfloat16 directly (pretrained weights).
        llm_embed = self.model.llm.get_input_embeddings()
        pad_embed_vec = llm_embed(self.model._pad_id_buf).detach().squeeze(0)  # [4096]
        projected = projected + pad_embed_vec.unsqueeze(0).unsqueeze(0)

        # --- Assemble prefix: [<mem>] [projected] [</mem>] (model.py:519-537) ---
        mem_start_emb = llm_embed(self.model._mem_id_buf).squeeze(0)   # [4096]
        mem_end_emb = llm_embed(self.model._mem_end_id_buf).squeeze(0) # [4096]

        # For B=1, no left-padding needed — directly concatenate
        prefix_embeds = torch.cat([
            mem_start_emb.unsqueeze(0),     # [1, 4096]
            projected.squeeze(0),           # [n_mem, 4096]
            mem_end_emb.unsqueeze(0),       # [1, 4096]
        ], dim=0).unsqueeze(0)              # [1, n_mem+2, 4096]

        prefix_mask = torch.ones(1, n_mem + 2, dtype=torch.long, device=self.device)

        self._memory_prefix_embeds = prefix_embeds
        self._memory_prefix_mask = prefix_mask
        logger.info("Memory encoded: %d tokens -> %d mem tokens", context_ids.shape[1], n_mem)

    def _build_llm_inputs(self, conversation_ids: list) -> dict:
        """Build inputs_embeds, attention_mask, position_ids for LLM generation.

        Replicates model.py forward() lines 557-581 for B=1.
        """
        llm_embed = self.model.llm.get_input_embeddings()
        target_ids = torch.tensor([conversation_ids], dtype=torch.long, device=self.device)
        target_embeds = llm_embed(target_ids)  # [1, T, 4096]

        if self._memory_prefix_embeds is not None:
            # Concat prefix + target (model.py:557)
            inputs_embeds = torch.cat(
                [self._memory_prefix_embeds, target_embeds], dim=1
            )
            attention_mask = torch.cat(
                [self._memory_prefix_mask, torch.ones_like(target_ids)], dim=1
            )
        else:
            # No memory: pure NTP path (model.py:479-486)
            inputs_embeds = target_embeds
            attention_mask = torch.ones_like(target_ids)

        # Build position_ids from attention mask (model.py:580-581)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)

        return {
            "inputs_embeds": inputs_embeds.to(dtype=self.model.llm.dtype),
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

    def _format_conversation(self, messages: list, template: str = "simple") -> str:
        """Format multi-turn messages into a single string for tokenization.

        Templates:
          - "none":   Raw concatenation, no role markers (matches pretraining).
          - "simple": ``User: ... \\n Assistant: ...`` (default).
          - "chatml": ``<|im_start|>role\\n...<|im_end|>`` (ChatML format).
        """
        if template == "none":
            parts = [msg["content"] for msg in messages]
            return "\n".join(parts) + "\n"

        if template == "chatml":
            parts = []
            for msg in messages:
                parts.append(
                    f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>"
                )
            parts.append("<|im_start|>assistant\n")
            return "\n".join(parts)

        # default: "simple"
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        parts.append("Assistant:")
        return "\n".join(parts)

    def _truncate_conversation(self, conversation_ids: list, max_len: int) -> list:
        """Truncate from the left (oldest turns) to fit within max_len."""
        prefix_len = (self._active_n_mem + 2) if self._memory_prefix_embeds is not None else 0
        available = max_len - prefix_len - 1  # reserve 1 for at least one generated token
        if len(conversation_ids) > available:
            conversation_ids = conversation_ids[-available:]
        return conversation_ids

    def _sample_next_token(self, logits, generated_ids, temperature,
                           top_p, top_k, repetition_penalty):
        """Sample a single token from logits."""
        scores = logits.clone()  # [1, vocab]

        # Repetition penalty
        if repetition_penalty != 1.0 and generated_ids:
            for tid in set(generated_ids):
                if scores[0, tid] < 0:
                    scores[0, tid] *= repetition_penalty
                else:
                    scores[0, tid] /= repetition_penalty

        # Temperature
        if temperature > 0:
            scores = scores / max(temperature, 1e-7)

        # Top-k
        if top_k > 0:
            topk_vals = torch.topk(scores, min(top_k, scores.size(-1)))[0]
            scores[scores < topk_vals[:, -1:]] = float("-inf")

        # Top-p (nucleus)
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(scores, descending=True)
            cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cum_probs > top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            indices_to_remove = sorted_idx[remove]
            scores[0, indices_to_remove] = float("-inf")

        # Sample or argmax
        if temperature > 0:
            probs = torch.softmax(scores, dim=-1)
            return torch.multinomial(probs, num_samples=1)  # [1, 1]
        return torch.argmax(scores, dim=-1, keepdim=True)   # [1, 1]

    @torch.no_grad()
    def generate(
        self,
        messages: list,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        stream: bool = False,
        template: str = "simple",
    ) -> Union[str, Iterator[str]]:
        """Generate a response given conversation messages.

        Uses manual prefill + autoregressive loop instead of HF generate()
        to avoid inputs_embeds compatibility issues across transformers versions.
        """
        self._stop_event.clear()
        conversation_text = self._format_conversation(messages, template=template)
        conversation_ids = self.tokenizer.encode(
            conversation_text, add_special_tokens=False
        )
        max_ctx = getattr(self.model.llm.config, "max_position_embeddings", 32768)
        conversation_ids = self._truncate_conversation(conversation_ids, max_len=max_ctx)

        llm_inputs = self._build_llm_inputs(conversation_ids)
        inputs_embeds = llm_inputs["inputs_embeds"]
        attention_mask = llm_inputs["attention_mask"]
        position_ids = llm_inputs["position_ids"]
        eos_id = self.tokenizer.eos_token_id

        # Phase 1: Prefill — single forward pass to build KV cache
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            prefill = self.model.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
                return_dict=True,
            )
        past_kv = prefill.past_key_values
        logits = prefill.logits[:, -1:, :]  # [1, 1, vocab]

        # Phase 2: Autoregressive decoding
        def _decode_loop(streamer=None):
            nonlocal past_kv, logits, attention_mask
            generated_ids = []
            try:
                for _ in range(max_new_tokens):
                    if self._stop_event.is_set():
                        break
                    next_token = self._sample_next_token(
                        logits.squeeze(1), generated_ids,
                        temperature, top_p, top_k, repetition_penalty,
                    )  # [1, 1]

                    tid = next_token.item()
                    if tid == eos_id:
                        break
                    generated_ids.append(tid)

                    if streamer is not None:
                        streamer.put(next_token)

                    # Extend attention mask
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones(1, 1, dtype=attention_mask.dtype,
                                   device=self.device),
                    ], dim=1)
                    new_pos = (attention_mask.sum(dim=-1, keepdim=True) - 1)

                    with torch.autocast(self.device.type, dtype=torch.bfloat16):
                        out = self.model.llm(
                            input_ids=next_token,
                            attention_mask=attention_mask,
                            position_ids=new_pos,
                            past_key_values=past_kv,
                            use_cache=True,
                            return_dict=True,
                        )
                    past_kv = out.past_key_values
                    logits = out.logits  # [1, 1, vocab]
            finally:
                if streamer is not None:
                    streamer.end()
            return generated_ids

        if stream:
            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True
            )
            thread = Thread(target=_decode_loop, args=(streamer,), daemon=True)
            thread.start()
            return streamer
        else:
            generated_ids = _decode_loop()
            return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

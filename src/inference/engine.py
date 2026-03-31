"""KawaiiLLM inference engine — faithful replica of training forward pass.

Loads a KawaiiLLM checkpoint (MemE + Projector + LLM) and provides:
  - set_memory(): encode context through MemE -> Projector, cache prefix embeddings
  - generate(): build LLM input (prefix + conversation) and generate with streaming

The encode/assembly logic replicates model.py encode_context() (lines 271-367)
and forward() non-NTP path (lines 488-589) exactly for B=1 inference.
"""

import logging
import os
from threading import Thread
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

        # Monkey-patch prepare_inputs_for_generation for inputs_embeds support
        self._patch_prepare_inputs()

        # Cached memory state
        self._memory_prefix_embeds: Optional[torch.Tensor] = None  # [1, n_mem+2, 4096]
        self._memory_prefix_mask: Optional[torch.Tensor] = None    # [1, n_mem+2]
        self._active_n_mem: int = 0

    def _patch_prepare_inputs(self):
        """Monkey-patch LLM's prepare_inputs_for_generation for inputs_embeds support.

        Following the C3.py:249-307 pattern. Qwen3's native method may not handle
        inputs_embeds correctly on the first generation step when no KV cache exists.
        """
        original = self.model.llm.prepare_inputs_for_generation

        def patched(input_ids, past_key_values=None, attention_mask=None,
                    inputs_embeds=None, **kwargs):
            # First step: use inputs_embeds if provided and no cache yet
            if inputs_embeds is not None and past_key_values is None:
                position_ids = None
                if attention_mask is not None:
                    position_ids = attention_mask.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attention_mask == 0, 1)
                return {
                    "inputs_embeds": inputs_embeds,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "past_key_values": past_key_values,
                    "use_cache": kwargs.get("use_cache", True),
                }
            # Subsequent steps: delegate to original (uses input_ids + KV cache)
            return original(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=None,  # force None so original uses input_ids
                **kwargs,
            )

        self.model.llm.prepare_inputs_for_generation = patched

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

        n_mem = n_mem or self.num_mem_tokens
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

    def _format_conversation(self, messages: list) -> str:
        """Format multi-turn messages into a single string for tokenization.

        Qwen3-8B-Base is a base model (no chat template), so we use a simple format.
        """
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        # Append prompt for assistant response
        parts.append("Assistant:")
        return "\n".join(parts)

    def _truncate_conversation(self, conversation_ids: list, max_len: int) -> list:
        """Truncate from the left (oldest turns) to fit within max_len."""
        prefix_len = (self._active_n_mem + 2) if self._memory_prefix_embeds is not None else 0
        available = max_len - prefix_len - 1  # reserve 1 for at least one generated token
        if len(conversation_ids) > available:
            conversation_ids = conversation_ids[-available:]
        return conversation_ids

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
    ) -> Union[str, Iterator[str]]:
        """Generate a response given conversation messages.

        Args:
            messages: List of {"role": "user"|"assistant", "content": str}.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            top_p: Nucleus sampling probability threshold.
            top_k: Top-k sampling.
            repetition_penalty: Repetition penalty (1.0 = no penalty).
            stream: If True, returns an Iterator yielding token strings.

        Returns:
            Generated text string, or an Iterator[str] if stream=True.
        """
        conversation_text = self._format_conversation(messages)
        conversation_ids = self.tokenizer.encode(
            conversation_text, add_special_tokens=False
        )
        max_ctx = getattr(self.model.llm.config, "max_position_embeddings", 32768)
        conversation_ids = self._truncate_conversation(conversation_ids, max_len=max_ctx)

        llm_inputs = self._build_llm_inputs(conversation_ids)

        gen_kwargs = {
            **llm_inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": max(temperature, 1e-7),
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        def _generate_with_autocast(**kw):
            with torch.autocast(self.device.type, dtype=torch.bfloat16):
                return self.model.llm.generate(**kw)

        if stream:
            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True
            )
            gen_kwargs["streamer"] = streamer

            # C2 fix: wrap streaming generation in autocast via helper
            thread = Thread(
                target=_generate_with_autocast, kwargs=gen_kwargs, daemon=True
            )
            thread.start()
            return streamer  # caller iterates: for token in streamer: ...
        else:
            output_ids = _generate_with_autocast(**gen_kwargs)
            # W3 fix: skip_prompt=True in TextIteratorStreamer handles stripping
            # for the streaming path. For non-streaming, HF generate() returns
            # the full sequence including a dummy input_ids when inputs_embeds
            # is used. Use the prompt length to strip correctly.
            prompt_len = llm_inputs["inputs_embeds"].shape[1]
            # HF may return output starting from prompt or from generated tokens
            # depending on version. Handle both cases:
            if output_ids.shape[1] > prompt_len:
                generated = output_ids[0, prompt_len:]
            else:
                generated = output_ids[0]
            return self.tokenizer.decode(generated, skip_special_tokens=True)

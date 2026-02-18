"""KawaiiLLM model: MemE (Qwen3-Embedding-4B) -> Projector -> LLM (Qwen3-8B-Base).

Plain nn.Module wrapping three components. Not a PreTrainedModel subclass to
avoid config registration complexity for a multi-model composition.

Architecture dimensions (dynamically read from model configs):
    - MemE (Qwen3-Embedding-4B): hidden_size = meme_hidden (2560)
    - LLM  (Qwen3-8B-Base):      hidden_size = llm_hidden  (4096)
    - Projector: RMSNorm(meme_hidden) -> Linear(meme_hidden, meme_hidden*4)
                 -> GELU -> Linear(meme_hidden*4, llm_hidden)
    - mem_embeddings: nn.Embedding(num_mem_tokens, meme_hidden)

Special tokens:
    - <mem> / </mem>: memory boundary markers (both MemE and LLM side)
    - <mempad>: placeholder (unused at runtime, reserved for tokenizer)
    - <AE>: auto-encoding mode signal (LLM side, reconstruction task only)
"""

import logging
import os
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100

# Special tokens registered by train.py
SPECIAL_TOKENS = ["<mem>", "<mempad>", "</mem>", "<AE>"]


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class KawaiiLLMModel(nn.Module):
    """KawaiiLLM: MemE encoder + Projector + LLM decoder."""

    def __init__(
        self,
        meme_model_name_or_path: str,
        llm_model_name_or_path: str,
        num_mem_tokens: int = 128,
        freeze_meme: bool = False,
        freeze_llm: bool = False,
        freeze_projector: bool = False,
    ):
        super().__init__()

        # Load MemE encoder
        logger.info("Loading MemE from %s", meme_model_name_or_path)
        self.meme = AutoModel.from_pretrained(
            meme_model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        meme_hidden = self.meme.config.hidden_size

        # Load LLM decoder
        logger.info("Loading LLM from %s", llm_model_name_or_path)
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        llm_hidden = self.llm.config.hidden_size

        # Learnable MEM tokens (appended to MemE input as Query Embeddings)
        self.mem_embeddings = nn.Embedding(num_mem_tokens, meme_hidden)
        nn.init.normal_(self.mem_embeddings.weight, std=0.02)

        # Projector: RMSNorm + 2-layer MLP with 4x expansion
        # Maps MemE output space -> LLM input space
        self.projector = nn.Sequential(
            RMSNorm(meme_hidden),
            nn.Linear(meme_hidden, meme_hidden * 4),
            nn.GELU(),
            nn.Linear(meme_hidden * 4, llm_hidden),
        )
        # Initialize projector Linear layers with small weights
        for module in self.projector:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)

        # Apply freezing
        if freeze_meme:
            self._freeze(self.meme)
        if freeze_llm:
            self._freeze(self.llm)
        if freeze_projector:
            self._freeze(self.projector)
            self.mem_embeddings.weight.requires_grad = False

        self._freeze_meme = freeze_meme
        self.num_mem_tokens = num_mem_tokens

        # Special token IDs (set by set_special_token_ids after tokenizer registration)
        self._mem_token_id: Optional[int] = None
        self._mem_end_token_id: Optional[int] = None
        self._ae_token_id: Optional[int] = None

        logger.info(
            "KawaiiLLM initialized: meme_hidden=%d, llm_hidden=%d, "
            "num_mem_tokens=%d, projector_expansion=%dx",
            meme_hidden,
            llm_hidden,
            num_mem_tokens,
            4,
        )

    def set_special_token_ids(self, token_ids: Dict[str, int]):
        """Set special token IDs after tokenizer registration.

        Args:
            token_ids: mapping from token string to ID, e.g.
                {"<mem>": 151936, "</mem>": 151938, "<AE>": 151939}
        """
        self._mem_token_id = token_ids["<mem>"]
        self._mem_end_token_id = token_ids["</mem>"]
        self._ae_token_id = token_ids["<AE>"]
        logger.info(
            "Special token IDs set: <mem>=%d, </mem>=%d, <AE>=%d",
            self._mem_token_id,
            self._mem_end_token_id,
            self._ae_token_id,
        )

    @staticmethod
    def _freeze(module: nn.Module):
        for p in module.parameters():
            p.requires_grad = False

    @property
    def config(self):
        """Return LLM config for HF Trainer compatibility."""
        return self.llm.config

    @property
    def dtype(self):
        return self.llm.dtype

    @property
    def device(self):
        return self.llm.device

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing on trainable components."""
        if not self._freeze_meme:
            self.meme.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )
        self.llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self):
        if not self._freeze_meme:
            self.meme.gradient_checkpointing_disable()
        self.llm.gradient_checkpointing_disable()

    def enable_input_require_grads(self):
        """Required when gradient checkpointing is used with frozen params."""
        self.meme.enable_input_require_grads()
        self.llm.enable_input_require_grads()

    def encode_context(
        self,
        context_ids: torch.LongTensor,
        context_attention_mask: torch.LongTensor,
        n_mem: int,
    ) -> torch.Tensor:
        """Encode context through MemE with <mem> Q_1..Q_n </mem> boundary tokens.

        Input layout: [text_tokens (left-padded)] [<mem>] [Q_1, ..., Q_n] [</mem>]

        Args:
            context_ids: [B, L] — tokenized context (left-padded).
            context_attention_mask: [B, L] — 1 for real tokens, 0 for padding.
            n_mem: number of MEM tokens to use this batch.

        Returns:
            mem_hidden: [B, n_mem, meme_hidden] — compressed memory vectors.
        """
        if self._mem_token_id is None:
            raise RuntimeError(
                "Special token IDs not set. Call set_special_token_ids() first."
            )

        B = context_ids.shape[0]
        device = context_ids.device
        meme_embed = self.meme.get_input_embeddings()

        # Get text embeddings from MemE's embedding layer
        text_embeds = meme_embed(context_ids)  # [B, L, meme_hidden]

        # Get <mem> and </mem> boundary token embeddings
        mem_start_id = torch.tensor(
            [self._mem_token_id], device=device, dtype=torch.long
        )
        mem_end_id = torch.tensor(
            [self._mem_end_token_id], device=device, dtype=torch.long
        )
        mem_start_emb = meme_embed(mem_start_id)  # [1, meme_hidden]
        mem_end_emb = meme_embed(mem_end_id)  # [1, meme_hidden]
        # Expand to batch: [B, 1, meme_hidden]
        mem_start_emb = mem_start_emb.unsqueeze(0).expand(B, -1, -1)
        mem_end_emb = mem_end_emb.unsqueeze(0).expand(B, -1, -1)

        # Get MEM token (Query) embeddings
        mem_embeds = self.mem_embeddings.weight[:n_mem]  # [n_mem, meme_hidden]
        mem_embeds = mem_embeds.unsqueeze(0).expand(B, -1, -1)  # [B, n_mem, meme_hidden]

        # Concatenate: [text, <mem>, Q_1..Q_n, </mem>]
        combined = torch.cat(
            [text_embeds, mem_start_emb, mem_embeds, mem_end_emb], dim=1
        )  # [B, L+1+n_mem+1, meme_hidden]

        # Extend attention mask: boundary + MEM tokens always attend
        extra_len = 1 + n_mem + 1  # <mem> + Q tokens + </mem>
        extra_mask = torch.ones(
            B, extra_len,
            dtype=context_attention_mask.dtype,
            device=device,
        )
        extended_mask = torch.cat(
            [context_attention_mask, extra_mask], dim=1
        )  # [B, L+1+n_mem+1]

        # Run through MemE — optionally without grad if frozen
        if self._freeze_meme:
            with torch.no_grad():
                outputs = self.meme(
                    inputs_embeds=combined,
                    attention_mask=extended_mask,
                )
        else:
            outputs = self.meme(
                inputs_embeds=combined,
                attention_mask=extended_mask,
            )

        # Extract MEM token hidden states: skip </mem> at -1, take n_mem before it
        # Layout: [..., <mem>, Q_1, ..., Q_n, </mem>]
        # Indices: ..., -(n_mem+2), -(n_mem+1), ..., -2, -1
        mem_hidden = outputs.last_hidden_state[:, -(n_mem + 1):-1, :]  # [B, n_mem, meme_hidden]

        if self._freeze_meme:
            # Detach to cut gradient flow, but re-enable grad for projector
            mem_hidden = mem_hidden.detach().requires_grad_(True)

        return mem_hidden

    def forward(
        self,
        context_ids: torch.LongTensor,
        context_attention_mask: torch.LongTensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        labels: torch.LongTensor,
        n_mem: int,
        task_type: str = "reconstruction",
        **kwargs,
    ):
        """Full forward pass: encode context -> project -> decode with LLM.

        LLM input layout:
            Reconstruction: [<mem>] [h_1, ..., h_N] [</mem>] [<AE>] [target_tokens...]
            Continuation:   [<mem>] [h_1, ..., h_N] [</mem>] [target_tokens...]

        Args:
            context_ids: [B, L_ctx] — context tokens (left-padded).
            context_attention_mask: [B, L_ctx] — context attention mask.
            input_ids: [B, T] — target tokens (right-padded).
            attention_mask: [B, T] — target attention mask.
            labels: [B, T] — target labels with IGNORE_INDEX for padding.
            n_mem: int — number of MEM tokens for this batch.
            task_type: "reconstruction" or "continuation".

        Returns:
            CausalLMOutputWithPast with loss.
        """
        B = input_ids.shape[0]
        device = input_ids.device
        llm_embed = self.llm.get_input_embeddings()

        # 1. Encode context through MemE
        mem_hidden = self.encode_context(
            context_ids, context_attention_mask, n_mem
        )  # [B, n_mem, meme_hidden]

        # 2. Project to LLM dimension
        projected = self.projector(mem_hidden)  # [B, n_mem, llm_hidden]

        # 3. Build LLM prefix: [<mem>, h_1..h_N, </mem>, (<AE>)?]
        mem_start_id = torch.tensor(
            [self._mem_token_id], device=device, dtype=torch.long
        )
        mem_end_id = torch.tensor(
            [self._mem_end_token_id], device=device, dtype=torch.long
        )
        mem_start_emb = llm_embed(mem_start_id).unsqueeze(0).expand(B, -1, -1)  # [B, 1, llm_hidden]
        mem_end_emb = llm_embed(mem_end_id).unsqueeze(0).expand(B, -1, -1)  # [B, 1, llm_hidden]

        prefix_parts = [mem_start_emb, projected, mem_end_emb]
        prefix_len = 1 + n_mem + 1  # <mem> + projected + </mem>

        if task_type == "reconstruction":
            ae_id = torch.tensor(
                [self._ae_token_id], device=device, dtype=torch.long
            )
            ae_emb = llm_embed(ae_id).unsqueeze(0).expand(B, -1, -1)  # [B, 1, llm_hidden]
            prefix_parts.append(ae_emb)
            prefix_len += 1

        prefix_embeds = torch.cat(prefix_parts, dim=1)  # [B, prefix_len, llm_hidden]

        # 4. Get target embeddings from LLM's embedding layer
        target_embeds = llm_embed(input_ids)  # [B, T, llm_hidden]

        # 5. Concatenate: [prefix, target_text]
        llm_input = torch.cat([prefix_embeds, target_embeds], dim=1)

        # 6. Build labels: IGNORE for prefix positions, real labels for target
        ignore_prefix = torch.full(
            (B, prefix_len), IGNORE_INDEX,
            dtype=labels.dtype, device=device,
        )
        llm_labels = torch.cat([ignore_prefix, labels], dim=1)

        # 7. Build attention mask for full sequence
        prefix_attn = torch.ones(
            B, prefix_len,
            dtype=attention_mask.dtype, device=device,
        )
        llm_attn_mask = torch.cat([prefix_attn, attention_mask], dim=1)

        # 8. Forward through LLM
        outputs = self.llm(
            inputs_embeds=llm_input,
            attention_mask=llm_attn_mask,
            labels=llm_labels,
        )

        return outputs

    def save_checkpoint(self, output_dir: str):
        """Save all components to separate subdirectories."""
        os.makedirs(output_dir, exist_ok=True)

        # Save MemE
        meme_dir = os.path.join(output_dir, "meme")
        self.meme.save_pretrained(meme_dir)
        logger.info("Saved MemE to %s", meme_dir)

        # Save LLM
        llm_dir = os.path.join(output_dir, "llm")
        self.llm.save_pretrained(llm_dir)
        logger.info("Saved LLM to %s", llm_dir)

        # Save projector + mem_embeddings
        projector_dir = os.path.join(output_dir, "projector")
        os.makedirs(projector_dir, exist_ok=True)
        torch.save(
            self.projector.state_dict(),
            os.path.join(projector_dir, "projector.pt"),
        )
        torch.save(
            self.mem_embeddings.state_dict(),
            os.path.join(projector_dir, "mem_embeddings.pt"),
        )
        logger.info("Saved projector + mem_embeddings to %s", projector_dir)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str,
        num_mem_tokens: int = 128,
        freeze_meme: bool = False,
        freeze_llm: bool = False,
        freeze_projector: bool = False,
    ) -> "KawaiiLLMModel":
        """Load model from a saved checkpoint directory."""
        meme_dir = os.path.join(checkpoint_dir, "meme")
        llm_dir = os.path.join(checkpoint_dir, "llm")
        projector_dir = os.path.join(checkpoint_dir, "projector")

        model = cls(
            meme_model_name_or_path=meme_dir,
            llm_model_name_or_path=llm_dir,
            num_mem_tokens=num_mem_tokens,
            freeze_meme=freeze_meme,
            freeze_llm=freeze_llm,
            freeze_projector=freeze_projector,
        )

        # Load projector + mem_embeddings weights
        projector_state = torch.load(
            os.path.join(projector_dir, "projector.pt"),
            map_location="cpu",
            weights_only=True,
        )
        model.projector.load_state_dict(projector_state)

        mem_state = torch.load(
            os.path.join(projector_dir, "mem_embeddings.pt"),
            map_location="cpu",
            weights_only=True,
        )
        model.mem_embeddings.load_state_dict(mem_state)

        logger.info("Loaded KawaiiLLM from checkpoint: %s", checkpoint_dir)
        return model

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
    - <mem> / </mem>: memory boundary markers (LLM side only)
    - <mempad>: placeholder (unused at runtime, reserved for tokenizer)
    - <AE>: auto-encoding mode signal (LLM side, reconstruction task only)
"""

import logging
import os
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        attn_implementation: Optional[str] = None,
    ):
        super().__init__()

        # Load MemE encoder.
        logger.info("Loading MemE from %s", meme_model_name_or_path)
        meme_kwargs = dict(
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        if attn_implementation:
            meme_kwargs["attn_implementation"] = attn_implementation
        self.meme = AutoModel.from_pretrained(
            meme_model_name_or_path,
            **meme_kwargs,
        )
        meme_hidden = self.meme.config.hidden_size

        # Load LLM decoder.
        logger.info("Loading LLM from %s", llm_model_name_or_path)
        llm_kwargs = dict(
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        if attn_implementation:
            llm_kwargs["attn_implementation"] = attn_implementation
        logger.info("Attention implementation: %s", attn_implementation or "default")
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name_or_path, **llm_kwargs,
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
        # Initialize projector Linear layers.
        # The LAST layer uses near-zero init so the projected prefix starts at
        # a similar scale to the LLM's own embeddings (~0.02 std).  Without
        # this, the projector output (std ≈ 1.4) is ~70× larger, causing
        # extreme attention scores, high loss, and NaN gradients.
        proj_linears = [m for m in self.projector if isinstance(m, nn.Linear)]
        for module in proj_linears[:-1]:
            nn.init.normal_(module.weight, std=0.02)
            nn.init.zeros_(module.bias)
        # Last layer: near-zero output so LLM sees almost no prefix initially
        nn.init.normal_(proj_linears[-1].weight, std=1e-4)
        nn.init.zeros_(proj_linears[-1].bias)

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
        self._pad_token_id: Optional[int] = None

        # Accumulator for per-task loss monitoring (read and cleared by TaskLossCallback)
        self._task_accum: Dict[int, list] = {0: [], 1: [], 2: []}

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
                {"<mem>": 151936, "</mem>": 151938, "<AE>": 151939,
                 "pad_token_id": 151643}
        """
        self._mem_token_id = token_ids["<mem>"]
        self._mem_end_token_id = token_ids["</mem>"]
        self._ae_token_id = token_ids["<AE>"]
        self._pad_token_id = token_ids["pad_token_id"]

        # Cache ID tensors as non-persistent buffers — auto-tracks device
        # with .to()/.cuda() but excluded from state_dict (derived values).
        self.register_buffer(
            "_mem_id_buf",
            torch.tensor([self._mem_token_id], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_mem_end_id_buf",
            torch.tensor([self._mem_end_token_id], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_pad_id_buf",
            torch.tensor([self._pad_token_id], dtype=torch.long),
            persistent=False,
        )

        logger.info(
            "Special token IDs set: <mem>=%d, </mem>=%d, <AE>=%d, pad=%s",
            self._mem_token_id,
            self._mem_end_token_id,
            self._ae_token_id,
            self._pad_token_id,
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
        """Enable gradient checkpointing on both MemE and LLM."""
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

    @torch.no_grad()
    def _store_task_info(
        self,
        logits: torch.Tensor,
        combined_labels: torch.LongTensor,
        n_mem: torch.LongTensor,
        input_ids: torch.LongTensor,
    ) -> None:
        """Compute per-sample losses and accumulate by task for monitoring.

        Task IDs: 0 = NTP, 1 = Reconstruction, 2 = Continuation.
        Called after each forward pass (training only); TaskLossCallback reads
        and clears _task_accum every logging_steps optimizer steps.
        """
        # Skip if enough samples already collected for this logging window.
        # Avoids running a full CE pass on every micro-batch; 32 samples give
        # stable per-task averages across logging_steps=10 / grad_accum=8.
        if sum(len(v) for v in self._task_accum.values()) >= 32:
            return

        B, L, V = logits.shape
        shift_logits = logits[:, :-1].contiguous()            # [B, L-1, V] — keep bf16
        shift_labels = combined_labels[:, 1:].contiguous()   # [B, L-1]

        per_token = F.cross_entropy(
            shift_logits.view(-1, V),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="none",
        ).view(B, -1)  # [B, L-1]

        valid = shift_labels != IGNORE_INDEX
        per_sample = (per_token * valid.float()).sum(-1) / valid.float().sum(-1).clamp(min=1)

        # Classify task per sample
        task_ids = n_mem.new_zeros(B)                               # 0 = NTP
        non_ntp = n_mem > 0
        task_ids[non_ntp] = 2                                       # 2 = Continuation
        ae_mask = non_ntp & (input_ids[:, 0] == self._ae_token_id)
        task_ids[ae_mask] = 1                                       # 1 = Reconstruction

        losses_cpu = per_sample.cpu()
        tasks_cpu = task_ids.cpu()
        for tid in range(3):
            mask = tasks_cpu == tid
            if mask.any():
                self._task_accum[tid].extend(losses_cpu[mask].tolist())

    def encode_context(
        self,
        context_ids: torch.LongTensor,
        context_attention_mask: torch.LongTensor,
        n_mem: int,
    ) -> torch.Tensor:
        """Encode context through MemE with MEM tokens appended at the end.

        Input layout: [text_tokens (left-padded)] [Q_1, ..., Q_n]

        No <mem>/</mem> boundary tokens on the MemE side. Qwen3-Embedding-4B
        uses causal attention + last-token pooling: the last position naturally
        aggregates the full sequence. Placing MEM tokens directly at the end
        lets Q_n occupy this privileged last position. Inserting </mem> after
        Q_n would waste that position (its hidden state is discarded).

        When called with per-sample n_mem, forward() passes max(n_mem) here.
        All Q tokens interact through MemE self-attention; forward() then
        selects the first n_mem[i] outputs per sample. This means Q tokens
        see more peers during training than at inference with exact n_mem.
        Accepted trade-off vs. the complexity of per-sample MemE attention masks.

        Args:
            context_ids: [B, L] — tokenized context (left-padded).
            context_attention_mask: [B, L] — 1 for real tokens, 0 for padding.
            n_mem: number of MEM tokens to use (max across batch).

        Returns:
            mem_hidden: [B, n_mem, meme_hidden] — compressed memory vectors.
        """
        B = context_ids.shape[0]
        device = context_ids.device
        meme_embed = self.meme.get_input_embeddings()

        # Get text embeddings from MemE's embedding layer
        text_embeds = meme_embed(context_ids)  # [B, L, meme_hidden]

        # Get MEM token (Query) embeddings
        mem_embeds = self.mem_embeddings.weight[:n_mem]  # [n_mem, meme_hidden]
        mem_embeds = mem_embeds.unsqueeze(0).expand(B, -1, -1)  # [B, n_mem, meme_hidden]

        # Concatenate: [text, Q_1..Q_n]
        # Q_n occupies the last position — equivalent to Qwen3-Embedding's last_token_pool.
        combined = torch.cat(
            [text_embeds, mem_embeds], dim=1
        )  # [B, L+n_mem, meme_hidden]

        # Extend attention mask: MEM tokens always attend
        extra_mask = torch.ones(
            B, n_mem,
            dtype=context_attention_mask.dtype,
            device=device,
        )
        extended_mask = torch.cat(
            [context_attention_mask, extra_mask], dim=1
        )  # [B, L+n_mem]

        # Prevent NaN from all-masked causal attention at left-padded positions.
        # Position 0 with mask=0 has NO valid attention targets under causal
        # attention (can only attend to itself, blocked by mask=0).  This makes
        # softmax output NaN.  Forward is unaffected (Q outputs never use
        # position 0), but backward computes `output * (grad - ...)` where
        # `output` is NaN → NaN gradient for ALL shared MemE attention params
        # (IEEE 754: 0 * NaN = NaN).  Setting mask[:, 0] = 1 lets position 0
        # self-attend; subsequent padded positions then attend to position 0,
        # preventing any all-masked rows.
        extended_mask[:, 0] = 1

        # Build position_ids from attention mask (handles left-padded context)
        position_ids = extended_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(extended_mask == 0, 0)

        # Run through MemE — optionally without grad if frozen
        if self._freeze_meme:
            with torch.no_grad():
                outputs = self.meme(
                    inputs_embeds=combined,
                    attention_mask=extended_mask,
                    position_ids=position_ids,
                )
        else:
            outputs = self.meme(
                inputs_embeds=combined,
                attention_mask=extended_mask,
                position_ids=position_ids,
            )

        # Extract the last n_mem positions — the MEM token hidden states.
        # Q_n (last position) has seen all context + all preceding Q tokens,
        # giving it the richest global aggregation (equivalent to last_token_pool).
        mem_hidden = outputs.last_hidden_state[:, -n_mem:, :]  # [B, n_mem, meme_hidden]

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
        n_mem: Union[int, torch.LongTensor],
        **kwargs,
    ):
        """Full forward pass: encode context -> project -> decode with LLM.

        Supports three task types via per-sample n_mem:
            - NTP (n_mem=0): pure language modeling, no MemE, no prefix.
            - Reconstruction/Continuation (n_mem>0): MemE + projected prefix.

        For non-NTP samples, the prefix is left-padded so that <mem>, valid
        latent tokens, and </mem> are right-aligned and contiguous.

        LLM input layout per sample:
            NTP:   [input_ids...]
            Other: [pad...] [<mem>] [h_1, ..., h_ni] [</mem>] [input_ids...]

        The <AE> task signal is embedded in input_ids by the dataset:
            Reconstruction: input_ids = [<AE>, target_tokens...]
            Continuation:   input_ids = [target_tokens...]
            NTP:            input_ids = [target_tokens...]

        Args:
            context_ids: [B, L_ctx] — context tokens (left-padded; minimal for NTP).
            context_attention_mask: [B, L_ctx] — context attention mask.
            input_ids: [B, T] — target tokens (right-padded, may start with <AE>).
            attention_mask: [B, T] — target attention mask.
            labels: [B, T] — target labels with IGNORE_INDEX for padding/<AE>.
            n_mem: [B] tensor or int — number of MEM tokens per sample (0 for NTP).

        Returns:
            CausalLMOutputWithPast with loss.
        """
        B = input_ids.shape[0]
        device = input_ids.device
        llm_embed = self.llm.get_input_embeddings()
        llm_hidden = self.llm.config.hidden_size

        # Normalize n_mem to a [B] tensor
        if isinstance(n_mem, int):
            n_mem = torch.full((B,), n_mem, dtype=torch.long, device=device)
        else:
            n_mem = n_mem.to(device)
        max_n_mem = n_mem.max().item()

        # Get target embeddings from LLM's embedding layer
        target_embeds = llm_embed(input_ids)  # [B, T, llm_hidden]

        if max_n_mem == 0:
            # --- Pure NTP batch ---
            if self.training:
                # ZeRO-2 requires gradient ALLREDUCE to fire in the same
                # bucket order on every rank.  Non-NTP ranks backward as
                # LLM → projector → MemE (sequential chain).  If we just
                # run the LLM here, MemE/projector hooks never fire, or
                # fire via an independent branch with a different order,
                # causing NCCL deadlock.
                #
                # Fix: run a minimal MemE + projector and embed the result
                # as a masked prefix in the LLM input.  Backward then
                # naturally flows LLM → prefix → projector → MemE —
                # matching the non-NTP path order exactly.
                # Explicit single-token dummy: guarantees MemE processes exactly
                # 2 tokens (pad + Q) regardless of batch content.
                dummy_ctx = self._pad_id_buf.view(1, 1)
                dummy_mask = torch.ones(
                    1, 1, device=device, dtype=context_attention_mask.dtype,
                )
                dummy_mem = self.encode_context(dummy_ctx, dummy_mask, 1)
                dummy_proj = self.projector(dummy_mem)  # [1, 1, llm_hidden]

                # 1-token prefix with mask=1, connected to dummy_proj for
                # grad flow.  mask=1 lets it self-attend (avoids NaN from
                # softmax over all-masked keys).  We also mask labels[0] so
                # no loss is computed at the prefix-to-target boundary where
                # logits are based on the dummy input.
                # Use pad token embedding instead of zeros.  Zero embeddings
                # cause RMSNorm backward to amplify gradients by 1/sqrt(eps)
                # ≈ 1000× per layer, overflowing to inf/NaN in bfloat16.
                prefix_embeds = llm_embed(self._pad_id_buf.view(1, 1)).detach().expand(B, -1, -1)
                prefix_embeds = prefix_embeds + (dummy_proj.sum() * 0.0)
                prefix_mask = attention_mask.new_ones(B, 1)

                combined_embeds = torch.cat([prefix_embeds, target_embeds], dim=1)
                combined_mask = torch.cat([prefix_mask, attention_mask], dim=1)
                prefix_labels = torch.full(
                    (B, 1), IGNORE_INDEX, dtype=labels.dtype, device=device,
                )
                safe_labels = labels.clone()
                safe_labels[:, 0] = IGNORE_INDEX
                combined_labels = torch.cat([prefix_labels, safe_labels], dim=1)
                position_ids = combined_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(combined_mask == 0, 0)

                ntp_outputs = self.llm(
                    inputs_embeds=combined_embeds,
                    attention_mask=combined_mask,
                    labels=combined_labels,
                    position_ids=position_ids,
                )
                self._store_task_info(ntp_outputs.logits, combined_labels, n_mem, input_ids)
                return ntp_outputs

            # Eval: no gradient sync needed, skip MemE entirely
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            return self.llm(
                inputs_embeds=target_embeds,
                attention_mask=attention_mask,
                labels=labels,
                position_ids=position_ids,
            )

        # --- Mixed or full MemE batch ---

        # 1. Encode context through MemE — only for non-NTP samples
        non_ntp_mask = n_mem > 0  # [B]
        if non_ntp_mask.all():
            mem_hidden = self.encode_context(
                context_ids, context_attention_mask, max_n_mem
            )
            projected = self.projector(mem_hidden)  # [B, max_n_mem, llm_hidden]
        else:
            # Only encode non-NTP samples to avoid wasting MemE compute
            mem_idx = non_ntp_mask.nonzero(as_tuple=True)[0]
            mem_hidden = self.encode_context(
                context_ids[mem_idx],
                context_attention_mask[mem_idx],
                max_n_mem,
            )
            projected_sub = self.projector(mem_hidden)
            projected = projected_sub.new_zeros(B, max_n_mem, projected_sub.size(-1))
            projected[mem_idx] = projected_sub

        # Pad token embedding for left-padded prefix positions and
        # residual base for projected tokens (prevents RMSNorm gradient explosion).
        pad_embed_vec = llm_embed(self._pad_id_buf).detach().squeeze(0)  # [llm_hidden]

        # Residual embedding: projector output starts near-zero (std ≈ 0.003),
        # causing RMSNorm backward to amplify gradients ~333× vs ~50× for normal
        # embeddings.  Adding pad embedding as constant base bounds amplification.
        projected = projected + pad_embed_vec.unsqueeze(0).unsqueeze(0)

        # Build per-sample left-padded prefix (using cached ID buffers)
        mem_start_emb = llm_embed(self._mem_id_buf).squeeze(0)      # [llm_hidden]
        mem_end_emb = llm_embed(self._mem_end_id_buf).squeeze(0)    # [llm_hidden]

        max_prefix_len = max_n_mem + 2  # <mem>(1) + max_n_mem + </mem>(1)

        # --- Vectorized prefix construction (no per-sample torch.cat/clone) ---
        # Initialize all positions with pad embedding (one allocation)
        prefix_embeds = pad_embed_vec.view(1, 1, -1).expand(B, max_prefix_len, -1).contiguous()

        non_ntp_idx = non_ntp_mask.nonzero(as_tuple=True)[0]

        if non_ntp_idx.numel() > 0:
            pad_lens = max_n_mem - n_mem  # [B], per-sample left-pad length

            # Place <mem> boundary at position pad_lens[i] for each non-NTP sample
            prefix_embeds[non_ntp_idx, pad_lens[non_ntp_idx]] = mem_start_emb

            # Place </mem> boundary at position max_n_mem+1 (same for all non-NTP)
            prefix_embeds[non_ntp_idx, max_n_mem + 1] = mem_end_emb

            # Place projected tokens (minimal loop: just slice assignment, no allocations)
            for i in non_ntp_idx:
                ni = n_mem[i].item()
                start = pad_lens[i].item() + 1
                prefix_embeds[i, start:start + ni] = projected[i, :ni]

        # Build attention mask vectorized: mask[i,j] = 1 iff j >= pad_lens[i] AND non-NTP
        pad_lens_all = torch.where(non_ntp_mask, max_n_mem - n_mem, max_prefix_len)  # NTP -> all masked
        positions = torch.arange(max_prefix_len, device=device).unsqueeze(0)  # [1, P]
        prefix_attn = (positions >= pad_lens_all.unsqueeze(1)).to(attention_mask.dtype)

        # Ensure position 0 of the prefix always has mask=1.  Without this,
        # left-padded positions (mask=0) have NO valid attention keys (causal
        # constraint), causing NaN in softmax → NaN hidden states → NaN grads
        # for shared parameters even when labels are IGNORE_INDEX.
        prefix_attn[:, 0] = 1

        # 3. Concatenate: [prefix, target_text]
        llm_input = torch.cat([prefix_embeds, target_embeds], dim=1)

        # 4. Build labels: IGNORE for all prefix positions, real labels for target.
        # For NTP samples (n_mem=0), also mask the first target token's label:
        # after HF's internal label shift, the last prefix position's logits
        # are evaluated against labels[:,0].  For NTP samples the prefix
        # logits are from zero/dummy input — masking avoids polluting the loss.
        safe_labels = labels.clone()
        ntp_mask = (n_mem == 0)
        if ntp_mask.any():
            safe_labels[ntp_mask, 0] = IGNORE_INDEX
        ignore_prefix = torch.full(
            (B, max_prefix_len), IGNORE_INDEX,
            dtype=labels.dtype, device=device,
        )
        llm_labels = torch.cat([ignore_prefix, safe_labels], dim=1)

        # 5. Build attention mask for full sequence
        llm_attn_mask = torch.cat([prefix_attn, attention_mask], dim=1)

        # 6. Build position_ids from attention mask (handles left-padded prefix)
        # NTP samples: prefix is all-masked, so position starts at target tokens.
        # Non-NTP: <mem> starts at position 0 regardless of padding.
        position_ids = llm_attn_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(llm_attn_mask == 0, 0)

        # 7. Forward through LLM
        outputs = self.llm(
            inputs_embeds=llm_input,
            attention_mask=llm_attn_mask,
            labels=llm_labels,
            position_ids=position_ids,
        )

        self._store_task_info(outputs.logits, llm_labels, n_mem, input_ids)
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
        attn_implementation: Optional[str] = None,
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
            attn_implementation=attn_implementation,
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

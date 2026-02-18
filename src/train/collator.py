"""Data collator for KawaiiLLM training.

Handles two different padding strategies:
    - context_ids: LEFT-padded (matching MemE's padding_side='left'),
      so MEM tokens are always at the rightmost positions.
    - input_ids / labels: RIGHT-padded (standard causal LM convention).

Also samples a single n_mem value per batch from a linear "many-to-few"
curriculum distribution, and passes through task_type.

Note: task_type is per-batch (required by model architecture since <AE> token
affects prefix length). With random shuffling, batches may contain mixed task
types; the majority type is used. A grouped batch sampler would eliminate this.
"""

import logging
import random
from collections import Counter
from typing import Dict, List

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


class KawaiiDataCollator:
    """Collates samples with left-padded context and right-padded target."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_mem_tokens_max: int = 128,
    ):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.num_mem_tokens_max = num_mem_tokens_max

        # Training progress (updated by CurriculumCallback)
        self._training_progress = 0.0

    def set_training_progress(self, progress: float):
        self._training_progress = max(0.0, min(1.0, progress))

    def _sample_n_mem(self) -> int:
        """Sample n_mem from linear 'many-to-few' curriculum distribution.

        Probability of each range changes linearly with progress p (0->1):
            P(high:  64-128) = 0.7 - 0.6*p   // 70% -> 10%
            P(mid:   16-64)  = 0.2 + 0.1*p   // 20% -> 30%
            P(low:    2-16)  = 0.1 + 0.2*p   // 10% -> 30%
            P(single:    1)  = 0.3*p          //  0% -> 30%
        """
        p = self._training_progress
        N = self.num_mem_tokens_max  # 128

        p_high = 0.7 - 0.6 * p    # P(64-N)
        p_mid = 0.2 + 0.1 * p     # P(16-64)
        p_low = 0.1 + 0.2 * p     # P(2-16)
        # p_single = 0.3 * p       # P(1) — implicit remainder

        r = random.random()
        if r < p_high:
            return random.randint(64, N)
        elif r < p_high + p_mid:
            return random.randint(16, 64)
        elif r < p_high + p_mid + p_low:
            return random.randint(2, 16)
        else:
            return 1

    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        # Sample n_mem for the entire batch
        n_mem = self._sample_n_mem()

        context_ids_list = [inst["context_ids"] for inst in instances]
        input_ids_list = [inst["input_ids"] for inst in instances]
        labels_list = [inst["labels"] for inst in instances]

        # Determine batch task_type via majority vote
        # Mixed task types occur because the DataLoader shuffles samples randomly.
        # Since <AE> prefix affects sequence layout, the batch must use a single type.
        task_types = [inst.get("task_type", "reconstruction") for inst in instances]
        counts = Counter(task_types)
        task_type = counts.most_common(1)[0][0]

        # --- Left-pad context_ids (for MemE, padding_side='left') ---
        max_ctx_len = max(ids.shape[0] for ids in context_ids_list)
        padded_ctx = []
        for ids in context_ids_list:
            pad_len = max_ctx_len - ids.shape[0]
            if pad_len > 0:
                padded_ctx.append(
                    F.pad(ids, (pad_len, 0), value=self.pad_token_id)
                )
            else:
                padded_ctx.append(ids)
        context_ids = torch.stack(padded_ctx)  # [B, L_ctx]
        # Build context attention mask from actual lengths (avoids EOS==PAD issue)
        context_attention_mask = torch.zeros_like(context_ids, dtype=torch.long)
        for i, ids in enumerate(context_ids_list):
            real_len = ids.shape[0]
            # Left-padded: real tokens are at the right end
            context_attention_mask[i, max_ctx_len - real_len:] = 1

        # --- Right-pad input_ids and labels (standard causal LM) ---
        max_tgt_len = max(ids.shape[0] for ids in input_ids_list)
        padded_ids = []
        padded_labels = []
        target_lengths = []
        for ids, lbl in zip(input_ids_list, labels_list):
            pad_len = max_tgt_len - ids.shape[0]
            target_lengths.append(ids.shape[0])
            if pad_len > 0:
                padded_ids.append(
                    F.pad(ids, (0, pad_len), value=self.pad_token_id)
                )
                padded_labels.append(
                    F.pad(lbl, (0, pad_len), value=IGNORE_INDEX)
                )
            else:
                padded_ids.append(ids)
                padded_labels.append(lbl)
        input_ids = torch.stack(padded_ids)  # [B, T]
        labels = torch.stack(padded_labels)  # [B, T]
        # Build attention mask from actual lengths (avoids EOS==PAD issue)
        attention_mask = torch.zeros_like(input_ids, dtype=torch.long)
        for i, real_len in enumerate(target_lengths):
            attention_mask[i, :real_len] = 1

        return {
            "context_ids": context_ids,
            "context_attention_mask": context_attention_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "n_mem": n_mem,
            "task_type": task_type,
        }

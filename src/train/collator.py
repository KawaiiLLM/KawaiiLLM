"""Data collator for KawaiiLLM training.

Handles two different padding strategies:
    - context_ids: LEFT-padded (matching MemE's padding_side='left'),
      so MEM tokens are always at the rightmost positions.
    - input_ids / labels: RIGHT-padded (standard causal LM convention).

n_mem is determined per-batch: all non-NTP samples in a batch share the
same n_mem value (uniform [1, num_mem_tokens]).  This ensures:
    1. Zero MemE compute waste (all MEM hidden states are used).
    2. Every non-NTP sample's last MEM token occupies the true last
       position in the MemE sequence, inheriting last-token-pool
       aggregation semantics from Qwen3-Embedding's causal attention.

Task type (reconstruction vs continuation) is handled per-sample in the
dataset via <AE> token in input_ids — no per-batch task_type needed.
"""

import random
from typing import Dict, List

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer

IGNORE_INDEX = -100


class KawaiiDataCollator:
    """Collates samples with left-padded context and right-padded target."""

    def __init__(self, tokenizer: PreTrainedTokenizer, num_mem_tokens: int = 128):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.num_mem_tokens = num_mem_tokens

    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        context_ids_list = [inst["context_ids"] for inst in instances]
        input_ids_list = [inst["input_ids"] for inst in instances]
        labels_list = [inst["labels"] for inst in instances]
        n_mem_list = [inst["n_mem"] for inst in instances]

        # --- Batch-level n_mem: all non-NTP samples share one value ---
        # Non-NTP samples (n_mem > 0) get a single randomly sampled n_mem.
        # This ensures all MEM tokens in encode_context are used (zero waste)
        # and every sample's last Q token occupies the true last position.
        has_non_ntp = any(nm > 0 for nm in n_mem_list)
        if has_non_ntp:
            batch_n_mem = random.randint(1, self.num_mem_tokens)
            n_mem_list = [batch_n_mem if nm > 0 else 0 for nm in n_mem_list]

        n_mem = torch.tensor(n_mem_list, dtype=torch.long)  # [B]

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
        # Note: reconstruction samples have <AE> prepended (longer by 1 token),
        # continuation samples do not. Padding handles the length difference.
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
        }

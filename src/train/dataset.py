"""KawaiiDataset: byte-offset indexed dataset with curriculum sampling.

Two training tasks:
    1. Reconstruction: context -> MemE -> latent -> LLM -> same text
    2. Continuation: split_x -> MemE -> latent -> LLM -> split_{x+1}

Curriculum learning controls n_mem distribution and task mix based on
training progress (set externally by CurriculumCallback).
"""

import json
import logging
import random
from typing import Dict, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


class KawaiiDataset(Dataset):
    """Dataset with byte-offset random access and curriculum sampling."""

    def __init__(
        self,
        index_path: str,
        tokenizer: PreTrainedTokenizer,
        context_max_length: int = 4096,
        target_max_length: int = 4096,
    ):
        logger.info("Loading index from %s", index_path)
        with open(index_path, "r") as f:
            index = json.load(f)

        self.entries = index["entries"]
        self.continuation_pairs = index.get("continuation_pairs", [])
        self.tokenizer = tokenizer
        self.context_max_length = context_max_length
        self.target_max_length = target_max_length

        # Build set of entries that have continuation targets
        self._continuation_source_set = set()
        self._continuation_map: Dict[int, int] = {}
        for src_idx, tgt_idx in self.continuation_pairs:
            self._continuation_source_set.add(src_idx)
            self._continuation_map[src_idx] = tgt_idx

        # Training progress for curriculum (updated by callback)
        self._training_progress = 0.0

        # File handle cache (per-worker, safe with DataLoader fork)
        self._file_handles: Dict[str, object] = {}

        logger.info(
            "Dataset: %d entries, %d continuation pairs",
            len(self.entries),
            len(self.continuation_pairs),
        )

    def __del__(self):
        """Close cached file handles."""
        for fh in self._file_handles.values():
            try:
                fh.close()
            except Exception:
                pass
        self._file_handles.clear()

    @staticmethod
    def worker_init_fn(worker_id: int):
        """DataLoader worker_init_fn: reset file handles and seed RNG per worker."""
        import torch.utils.data as data
        worker_info = data.get_worker_info()
        if worker_info is not None:
            dataset = worker_info.dataset
            # Clear file handles so each worker opens its own
            dataset._file_handles = {}
            # Seed random per worker for curriculum sampling diversity
            random.seed(worker_info.seed % (2**32))

    def set_training_progress(self, progress: float):
        """Set current training progress [0, 1] for curriculum sampling."""
        self._training_progress = max(0.0, min(1.0, progress))

    def _get_file_handle(self, filepath: str):
        # Use binary mode to match byte offsets from build_index.py (which uses "rb")
        if filepath not in self._file_handles:
            self._file_handles[filepath] = open(filepath, "rb")
        return self._file_handles[filepath]

    def _read_entry(self, idx: int) -> dict:
        """Read a single entry by byte offset."""
        entry = self.entries[idx]
        fh = self._get_file_handle(entry["file"])
        fh.seek(entry["offset"])
        line = fh.readline()
        return json.loads(line.decode("utf-8"))

    def _get_task_type(self, idx: int) -> str:
        """Determine task type: deterministic assignment.

        - Entries without continuation pairs always do reconstruction.
        - During warmup (progress < 10%), all entries do reconstruction.
        - After warmup, entries with continuation pairs always do continuation.
        """
        if idx not in self._continuation_source_set:
            return "reconstruction"
        if self._training_progress < 0.1:
            return "reconstruction"
        return "continuation"

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        """Return a single training sample.

        Returns:
            dict with:
                context_ids: [L_ctx] — tokenized context for MemE
                input_ids: [T] — tokenized target for LLM
                labels: [T] — labels (same as input_ids, padding will be IGNORE)
                task_type: "reconstruction" or "continuation"
        """
        task_type = self._get_task_type(idx)
        record = self._read_entry(idx)
        text = record.get("text", "")

        if task_type == "continuation" and idx in self._continuation_map:
            # Context = current split, target = next split
            target_idx = self._continuation_map[idx]
            target_record = self._read_entry(target_idx)
            context_text = text
            target_text = target_record.get("text", "")
        else:
            # Reconstruction: context and target are the same text
            context_text = text
            target_text = text

        # Tokenize context (for MemE) — no special tokens needed
        context_ids = self.tokenizer.encode(
            context_text,
            add_special_tokens=False,
            max_length=self.context_max_length,
            truncation=True,
        )

        # Tokenize target (for LLM)
        target_ids = self.tokenizer.encode(
            target_text,
            add_special_tokens=False,
            max_length=self.target_max_length,
            truncation=True,
        )

        # For causal LM, labels = input_ids (shift is done internally by the model)
        context_ids = torch.tensor(context_ids, dtype=torch.long)
        input_ids = torch.tensor(target_ids, dtype=torch.long)
        labels = input_ids.clone()

        return {
            "context_ids": context_ids,
            "input_ids": input_ids,
            "labels": labels,
            "task_type": task_type,
        }

"""KawaiiDataset: byte-offset indexed dataset with curriculum sampling.

Two training tasks:
    1. Reconstruction (AE): context -> MemE -> latent -> LLM (<AE> signal) -> same text
    2. Continuation (AR): split_x -> MemE -> latent -> LLM -> split_{x+1}

Task assignment:
    - Continuation probability linearly increases from 0% to 50% over the first
      10% of training, then stays at 50%.
    - If continuation is chosen, use the natural next chunk if available;
      otherwise, split the current text at a newline boundary.
    - <AE> token is prepended to input_ids for reconstruction tasks, so each
      sample independently carries its own task signal (no per-batch constraint).
"""

import json
import logging
import random
from typing import Dict, List, Optional, Tuple

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

        # <AE> token ID for reconstruction task signal
        self._ae_token_id = tokenizer.convert_tokens_to_ids("<AE>")

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
            "Dataset: %d entries, %d continuation pairs, <AE> id=%d",
            len(self.entries),
            len(self.continuation_pairs),
            self._ae_token_id,
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
        """Determine task type with linearly increasing continuation probability.

        Continuation probability:
            - progress < 0.1: linearly ramps from 0% to 50%
            - progress >= 0.1: stays at 50%

        If continuation is chosen but no natural next chunk exists,
        the text will be split synthetically (handled in __getitem__).
        """
        p = self._training_progress
        if p < 0.1:
            cont_prob = 0.5 * (p / 0.1)  # 0% -> 50%
        else:
            cont_prob = 0.5

        if random.random() < cont_prob:
            return "continuation"
        return "reconstruction"

    # Minimum characters per half when splitting for synthetic continuation.
    # Texts shorter than 2x this or without a valid \n split point fall back
    # to reconstruction.
    _MIN_SPLIT_CHARS = 64

    @staticmethod
    def _split_text(text: str, min_chars: int = 64) -> Optional[Tuple[str, str]]:
        """Try to split text at a newline for synthetic continuation.

        Returns (context, target) if a good split is found, None otherwise.
        A "good split" requires:
            1. The text contains at least one ``\\n``.
            2. Both halves are at least *min_chars* characters long.

        Among valid newlines, picks the one closest to the midpoint.
        """
        if len(text) < min_chars * 2:
            return None

        mid = len(text) // 2

        # Find newline positions that leave both halves >= min_chars
        candidates = [
            i for i, c in enumerate(text)
            if c == "\n" and i >= min_chars and len(text) - i - 1 >= min_chars
        ]

        if not candidates:
            return None

        best = min(candidates, key=lambda pos: abs(pos - mid))
        return text[:best + 1], text[best + 1:]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        """Return a single training sample.

        For reconstruction: input_ids starts with <AE> token as task signal.
        For continuation: input_ids has no prefix (target text only).

        Returns:
            dict with:
                context_ids: [L_ctx] — tokenized context for MemE
                input_ids: [T] — tokenized target for LLM (may start with <AE>)
                labels: [T] — labels (IGNORE for <AE> position, real for rest)
        """
        task_type = self._get_task_type(idx)
        record = self._read_entry(idx)
        text = record.get("text", "")

        if task_type == "continuation":
            if idx in self._continuation_map:
                # Natural continuation: context = this chunk, target = next chunk
                target_idx = self._continuation_map[idx]
                target_record = self._read_entry(target_idx)
                context_text = text
                target_text = target_record.get("text", "")
            else:
                # Synthetic continuation: try to split at a newline boundary
                split = self._split_text(text, self._MIN_SPLIT_CHARS)
                if split is not None:
                    context_text, target_text = split
                else:
                    # Text too short or no valid \n — fall back to reconstruction
                    task_type = "reconstruction"
                    context_text = text
                    target_text = text

        if task_type == "reconstruction":
            context_text = text
            target_text = text

        # Tokenize context (for MemE)
        context_ids = self.tokenizer.encode(
            context_text,
            add_special_tokens=False,
            max_length=self.context_max_length,
            truncation=True,
        )

        # Tokenize target (for LLM)
        # Reserve 1 token for EOS (and 1 more for <AE> in reconstruction)
        ae_overhead = 1 if task_type == "reconstruction" else 0
        target_ids = self.tokenizer.encode(
            target_text,
            add_special_tokens=False,
            max_length=self.target_max_length - 1 - ae_overhead,
            truncation=True,
        )
        # Append EOS so the model learns when to stop generating
        target_ids = target_ids + [self.tokenizer.eos_token_id]

        if task_type == "reconstruction":
            # Prepend <AE> token as reconstruction signal
            # Label for <AE> is IGNORE (it's a provided signal, not a prediction target)
            input_ids = torch.tensor(
                [self._ae_token_id] + target_ids, dtype=torch.long
            )
            labels = torch.tensor(
                [IGNORE_INDEX] + target_ids, dtype=torch.long
            )
        else:
            input_ids = torch.tensor(target_ids, dtype=torch.long)
            labels = input_ids.clone()

        context_ids = torch.tensor(context_ids, dtype=torch.long)

        return {
            "context_ids": context_ids,
            "input_ids": input_ids,
            "labels": labels,
        }

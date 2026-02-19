"""KawaiiDataset: byte-offset indexed dataset with deterministic task rotation.

Three training tasks:
    1. NTP (Next Token Prediction): pure language modeling, no MemE involvement.
    2. Reconstruction (AE): context -> MemE -> latent -> LLM (<AE> signal) -> same text.
    3. Continuation (AR): split_x -> MemE -> latent -> LLM -> split_{x+1}.

Task assignment:
    - Deterministic rotation: each sample is assigned exactly one task per epoch.
      Over every consecutive 3 epochs, each sample trains with each task exactly
      once. Formula: task_idx = (sample_idx + epoch + epoch // 3) % 3.
    - Equal 1/3 ratio from the start, no warmup needed.
    - If continuation is assigned but cannot be executed (text too short to split,
      no natural next chunk), falls back to NTP.
    - <AE> token is prepended to input_ids for reconstruction tasks.

Per-sample n_mem:
    - NTP: n_mem = 0 (no MemE involvement).
    - Reconstruction / Continuation: uniform [1, num_mem_tokens].
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

TASK_TYPES = ["ntp", "reconstruction", "continuation"]


class KawaiiDataset(Dataset):
    """Dataset with byte-offset random access and deterministic task rotation."""

    def __init__(
        self,
        index_path: str,
        tokenizer: PreTrainedTokenizer,
        context_max_length: int = 4096,
        target_max_length: int = 4096,
        num_mem_tokens: int = 128,
    ):
        logger.info("Loading index from %s", index_path)
        with open(index_path, "r") as f:
            index = json.load(f)

        self.entries = index["entries"]
        self.continuation_pairs = index.get("continuation_pairs", [])
        self.tokenizer = tokenizer
        self.context_max_length = context_max_length
        self.target_max_length = target_max_length
        self.num_mem_tokens = num_mem_tokens

        # <AE> token ID for reconstruction task signal
        self._ae_token_id = tokenizer.convert_tokens_to_ids("<AE>")

        # Build map: source_idx -> target_idx for continuation pairs
        self._continuation_map: Dict[int, int] = {}
        for src_idx, tgt_idx in self.continuation_pairs:
            self._continuation_map[src_idx] = tgt_idx

        # Current epoch for deterministic task rotation (updated by callback)
        self._current_epoch = 0

        # File handle cache (per-worker, safe with DataLoader fork)
        self._file_handles: Dict[str, object] = {}

        logger.info(
            "Dataset: %d entries, %d continuation pairs, <AE> id=%d, "
            "num_mem_tokens=%d",
            len(self.entries),
            len(self.continuation_pairs),
            self._ae_token_id,
            num_mem_tokens,
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
            # Seed random per worker for n_mem sampling diversity
            random.seed(worker_info.seed % (2**32))

    def set_current_epoch(self, epoch: int):
        """Set current epoch for deterministic task rotation."""
        self._current_epoch = epoch

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
        """Deterministic task assignment based on sample index and epoch.

        Uses modular arithmetic to guarantee:
            - Each sample trains with each task exactly once per 3-epoch cycle.
            - Within any single epoch, tasks are perfectly balanced (1/3 each).
            - The epoch // 3 shift varies the starting task across cycles.
        """
        epoch = self._current_epoch
        task_idx = (idx + epoch + epoch // 3) % 3
        return TASK_TYPES[task_idx]

    def _sample_n_mem(self, task_type: str) -> int:
        """Sample n_mem based on task type.

        NTP: 0 (no MemE involvement).
        Reconstruction / Continuation: uniform [1, num_mem_tokens].
        """
        if task_type == "ntp":
            return 0
        return random.randint(1, self.num_mem_tokens)

    # Minimum characters per half when splitting for synthetic continuation.
    # Texts shorter than 2x this or without a valid \n split point fall back
    # to NTP.
    _MIN_SPLIT_CHARS = 256

    @staticmethod
    def _split_text(text: str, min_chars: int = 256) -> Optional[Tuple[str, str]]:
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

    def _build_ntp_sample(self, text: str) -> dict:
        """Build a pure NTP sample (no MemE involvement)."""
        target_ids = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            max_length=self.target_max_length - 1,  # reserve for EOS
            truncation=True,
        )
        target_ids = target_ids + [self.tokenizer.eos_token_id]
        input_ids = torch.tensor(target_ids, dtype=torch.long)
        labels = input_ids.clone()
        # Minimal context placeholder (ignored since n_mem=0)
        context_ids = torch.tensor(
            [self.tokenizer.pad_token_id], dtype=torch.long
        )
        return {
            "context_ids": context_ids,
            "input_ids": input_ids,
            "labels": labels,
            "n_mem": 0,
        }

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        """Return a single training sample.

        For NTP: pure language modeling, no MemE, n_mem=0.
        For reconstruction: input_ids starts with <AE> token as task signal.
        For continuation: input_ids has no prefix (target text only).

        Returns:
            dict with:
                context_ids: [L_ctx] — tokenized context for MemE (minimal for NTP)
                input_ids: [T] — tokenized target for LLM (may start with <AE>)
                labels: [T] — labels (IGNORE for <AE> position, real for rest)
                n_mem: int — number of latent tokens (0 for NTP)
        """
        task_type = self._get_task_type(idx)
        record = self._read_entry(idx)
        text = record.get("text", "")

        # --- NTP path ---
        if task_type == "ntp":
            return self._build_ntp_sample(text)

        # --- Continuation path ---
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
                    # Text too short or no valid \n — fall back to NTP
                    return self._build_ntp_sample(text)

        # --- Reconstruction path ---
        if task_type == "reconstruction":
            context_text = text
            target_text = text

        # --- Shared tokenization for reconstruction / continuation ---
        # Tokenize context (for MemE)
        context_ids = self.tokenizer.encode(
            context_text,
            add_special_tokens=False,
            max_length=self.context_max_length,
            truncation=True,
        )

        # Determine EOS policy:
        # - Reconstruction: always add EOS
        # - Continuation: check if TARGET chunk has further continuation
        #   - target has next chunk: no EOS
        #   - target is last chunk / synthetic split: add EOS
        add_eos = True
        if task_type == "continuation" and idx in self._continuation_map:
            tgt_idx = self._continuation_map[idx]
            if tgt_idx in self._continuation_map:
                add_eos = False

        # Tokenize target (for LLM)
        # Reserve 1 token for EOS (if needed) and 1 more for <AE> in reconstruction
        reserved = (1 if add_eos else 0) + (1 if task_type == "reconstruction" else 0)
        target_ids = self.tokenizer.encode(
            target_text,
            add_special_tokens=False,
            max_length=self.target_max_length - reserved,
            truncation=True,
        )
        # Append EOS conditionally
        if add_eos:
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

        # Sample n_mem based on final task_type
        n_mem = self._sample_n_mem(task_type)

        return {
            "context_ids": context_ids,
            "input_ids": input_ids,
            "labels": labels,
            "n_mem": n_mem,
        }

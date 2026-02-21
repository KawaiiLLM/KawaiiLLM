"""KawaiiDataset: byte-offset indexed dataset with deterministic task rotation.

Three training tasks:
    1. NTP (Next Token Prediction): pure language modeling, no MemE involvement.
    2. Reconstruction (AE): context -> MemE -> latent -> LLM (<AE> signal) -> same text.
    3. Continuation (AR): prev_chunk -> MemE -> latent -> LLM -> current_chunk.

Task assignment (predecessor-based 2-task rotation):
    - Entries WITH a predecessor chunk: 2-task rotation over reconstruction,
      continuation.  Formula: task_idx = (idx + epoch) % 2
    - Entries WITHOUT a predecessor chunk: 2-task rotation over NTP,
      reconstruction.  Formula: task_idx = (idx + epoch) % 2
    - Continuation uses the predecessor as MemE context and the current chunk
      as LLM target, matching the inference scenario where compressed history
      is fed to the LLM to generate the next response.

Per-sample n_mem:
    - NTP: n_mem = 0 (no MemE involvement).
    - Reconstruction / Continuation: uniform [1, num_mem_tokens].

EOS policy:
    EOS is added only when the text genuinely ends. Merged entries (artificial
    \\n\\n joins) never get EOS. Reconstruction always gets EOS on non-merged
    entries (complete text boundary). NTP and continuation check has_next.

    NTP (only first-of-document chunks):
        merged entry             -> no EOS  (artificial boundary)
        non-merged, has next     -> no EOS  (text continues)
        non-merged, no next      -> EOS     (document ends)
    Reconstruction:
        merged entry             -> no EOS  (artificial boundary)
        non-merged               -> EOS     (complete reconstruction)
    Continuation (current chunk is LLM target):
        current has next chunk   -> no EOS  (text continues)
        current is last chunk    -> EOS     (document ends)

Merged chunks:
    - Short orphan entries merged by build_index.py have a "parts" field
      instead of "file"/"offset". These are read by loading each part and
      joining with "\\n\\n".
"""

import json
import logging
import random
from typing import Dict

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100

TASK_TYPES_NO_PREV = ["ntp", "reconstruction"]
TASK_TYPES_HAS_PREV = ["reconstruction", "continuation"]


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

        # Build map: target_idx -> source_idx (predecessor lookup for continuation)
        self._prev_map: Dict[int, int] = {}
        for src_idx, tgt_idx in self.continuation_pairs:
            self._prev_map[tgt_idx] = src_idx

        # Sets for task assignment and EOS determination
        self._has_prev: set = set(self._prev_map.keys())
        self._has_next: set = {src for src, _ in self.continuation_pairs}

        # Current epoch for deterministic task rotation (updated by callback)
        self._current_epoch = 0

        # File handle cache (per-worker, safe with DataLoader fork)
        self._file_handles: Dict[str, object] = {}

        n_has_prev = len(self._has_prev)
        n_no_prev = len(self.entries) - n_has_prev
        n_merged = sum(1 for e in self.entries if "parts" in e)
        logger.info(
            "Dataset: %d entries (%d has-prev recon/cont, %d no-prev NTP/recon, "
            "%d merged), %d continuation pairs, <AE> id=%d, num_mem_tokens=%d",
            len(self.entries), n_has_prev, n_no_prev, n_merged,
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

    def _read_record(self, file: str, offset: int) -> dict:
        """Read a single JSONL record by file path and byte offset."""
        fh = self._get_file_handle(file)
        fh.seek(offset)
        line = fh.readline()
        return json.loads(line.decode("utf-8"))

    def _read_entry(self, idx: int) -> dict:
        """Read entry text. Handles both normal and merged entries."""
        entry = self.entries[idx]

        if "parts" in entry:
            # Merged entry: read each part and join with \n\n
            texts = []
            for part in entry["parts"]:
                record = self._read_record(part["file"], part["offset"])
                texts.append(record.get("text", ""))
            return {"text": "\n\n".join(t for t in texts if t)}

        return self._read_record(entry["file"], entry["offset"])

    def _is_merged(self, idx: int) -> bool:
        """Check if entry is a merged chunk (artificial \n\n boundary)."""
        return "parts" in self.entries[idx]

    def _get_task_type(self, idx: int) -> str:
        """Deterministic task assignment based on predecessor availability.

        Entries with a predecessor: 2-task rotation (reconstruction,
        continuation). Each sample alternates between the two every epoch.

        Entries without a predecessor: 2-task rotation (NTP, reconstruction).
        Each sample alternates between the two every epoch.
        """
        epoch = self._current_epoch
        if idx in self._has_prev:
            task_idx = (idx + epoch) % 2
            return TASK_TYPES_HAS_PREV[task_idx]
        else:
            task_idx = (idx + epoch) % 2
            return TASK_TYPES_NO_PREV[task_idx]

    def _sample_n_mem(self, task_type: str) -> int:
        """Sample n_mem based on task type.

        NTP: 0 (no MemE involvement).
        Reconstruction / Continuation: uniform [1, num_mem_tokens].
        """
        if task_type == "ntp":
            return 0
        return random.randint(1, self.num_mem_tokens)

    def _should_add_eos(self, idx: int, task_type: str) -> bool:
        """Determine whether to append EOS token.

        - Merged entries: never (artificial boundary from chunk merging).
        - NTP non-merged, has next: no EOS (document continues).
        - NTP non-merged, no next: EOS (document ends).
        - Reconstruction non-merged: always EOS (complete reconstruction).
        - Continuation: current chunk is LLM target. EOS if document ends here.
        """
        if self._is_merged(idx):
            return False
        if task_type == "reconstruction":
            return True
        if task_type == "ntp":
            return idx not in self._has_next
        if task_type == "continuation":
            # Current chunk is the LLM target; EOS if document ends here
            return idx not in self._has_next
        raise ValueError(f"Unknown task type: {task_type}")

    def _build_ntp_sample(self, text: str, add_eos: bool) -> dict:
        """Build a pure NTP sample (no MemE involvement)."""
        reserved = 1 if add_eos else 0
        target_ids = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            max_length=self.target_max_length - reserved,
            truncation=True,
        )
        if add_eos:
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
        For continuation: predecessor -> MemE context, current chunk -> LLM target.

        Returns:
            dict with:
                context_ids: [L_ctx] -- tokenized context for MemE (minimal for NTP)
                input_ids: [T] -- tokenized target for LLM (may start with <AE>)
                labels: [T] -- labels (IGNORE for <AE> position, real for rest)
                n_mem: int -- number of latent tokens (0 for NTP)
        """
        task_type = self._get_task_type(idx)
        record = self._read_entry(idx)
        text = record.get("text", "")
        add_eos = self._should_add_eos(idx, task_type)

        # --- NTP path ---
        if task_type == "ntp":
            return self._build_ntp_sample(text, add_eos=add_eos)

        # --- Continuation path (predecessor -> MemE, current -> LLM) ---
        if task_type == "continuation":
            prev_idx = self._prev_map[idx]
            prev_record = self._read_entry(prev_idx)
            context_text = prev_record.get("text", "")
            target_text = text

        # --- Reconstruction path ---
        elif task_type == "reconstruction":
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

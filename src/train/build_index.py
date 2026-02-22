"""Standalone script to build a byte-offset index over formatted JSONL files.

Scans all JSONL files in the given directories, records byte offset per line,
identifies continuation pairs, optionally upsamples small sources, and merges
short orphan chunks.

Supports train/val split via --val_ratio (default 2%). The split is done at
the document level (by source+id) so all chunks of the same document stay in
the same split. Val set skips upsample and merge to preserve raw distribution.

Usage:
    python src/train/build_index.py \
        --data_dirs data/novels/formatted data/bilibili/formatted ... \
        --output_path data/train_index.json \
        --val_ratio 0.02 --val_output_path data/val_index.json \
        --upsample moegirl:3 \
        --merge_max_tokens 3500 --merge_short_threshold 2048
"""

import argparse
import json
import logging
import os
import random
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def scan_jsonl_file(filepath: str) -> list:
    """Scan a single JSONL file and return entries with byte offsets."""
    entries = []
    # Fallback source from directory name (two levels up: data/{source}/formatted/)
    dir_source = os.path.basename(os.path.dirname(os.path.dirname(filepath)))

    with open(filepath, "rb") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(
                    "Skipping invalid JSON at offset %d in %s", offset, filepath
                )
                continue

            entry = {
                "source": record.get("source", dir_source),
                "id": record.get("id", ""),
                "split": record.get("split", 0),
                "tokens": record.get("tokens", 0),
                "file": os.path.abspath(filepath),
                "offset": offset,
            }
            entries.append(entry)

    return entries


def build_continuation_pairs(entries: list) -> list:
    """Find continuation pairs: consecutive splits of the same document.

    A continuation pair (A, B) means B.split == A.split + 1 with the same
    source and id, so B can serve as the continuation target for A.
    """
    # Group by (source, id)
    groups = defaultdict(list)
    for idx, entry in enumerate(entries):
        key = (entry["source"], entry["id"])
        groups[key].append((entry["split"], idx))

    pairs = []
    for key, split_indices in groups.items():
        split_indices.sort(key=lambda x: x[0])
        for i in range(len(split_indices) - 1):
            split_a, idx_a = split_indices[i]
            split_b, idx_b = split_indices[i + 1]
            if split_b == split_a + 1:
                pairs.append([idx_a, idx_b])

    return pairs


def upsample_entries(entries: list, upsample_specs: list) -> list:
    """Upsample entries from specified sources by repeating them.

    Args:
        entries: list of index entries.
        upsample_specs: list of "source:ratio" strings, e.g. ["moegirl:3"].

    Returns:
        New entries list with upsampled sources repeated.
    """
    if not upsample_specs:
        return entries

    ratios = {}
    for spec in upsample_specs:
        parts = spec.rsplit(":", 1)
        if len(parts) != 2:
            logger.warning("Invalid upsample spec '%s', expected source:ratio", spec)
            continue
        source, ratio_str = parts
        try:
            ratio = int(ratio_str)
        except ValueError:
            logger.warning("Invalid ratio in upsample spec '%s'", spec)
            continue
        if ratio < 2:
            logger.warning("Upsample ratio must be >= 2, got %d for '%s'", ratio, source)
            continue
        ratios[source] = ratio

    if not ratios:
        return entries

    result = list(entries)
    for source, ratio in ratios.items():
        source_entries = [e for e in entries if e["source"] == source]
        if not source_entries:
            logger.warning("No entries found for source '%s', skipping upsample", source)
            continue
        # Each copy gets a unique id suffix so build_continuation_pairs
        # creates independent pairs per copy (not one giant group).
        for copy_num in range(1, ratio):
            for e in source_entries:
                new_entry = dict(e)
                new_entry["id"] = f"{e['id']}__up{copy_num}"
                result.append(new_entry)
        logger.info(
            "Upsampling '%s': %d original -> %d total (ratio %d)",
            source, len(source_entries), len(source_entries) * ratio, ratio,
        )

    return result


def merge_short_orphans(
    entries: list,
    continuation_pairs: list,
    merge_max_tokens: int = 3500,
    merge_short_threshold: int = 2048,
) -> list:
    """Merge short orphan chunks (no prev, no next) from the same source.

    Orphan = entry that is not the source or target of any continuation pair.
    Short orphan = orphan with tokens < merge_short_threshold.

    Merged entries replace their constituent entries and use a "parts" field
    instead of "file"/"offset". Non-orphans and long orphans are unchanged.

    Args:
        entries: list of index entries.
        continuation_pairs: list of [src_idx, tgt_idx] pairs.
        merge_max_tokens: max combined token count for a merged entry.
        merge_short_threshold: only merge orphans below this token count.

    Returns:
        New entries list with short orphans merged.
    """
    # Identify indices that participate in continuation pairs
    has_next = set()
    has_prev = set()
    for src_idx, tgt_idx in continuation_pairs:
        has_next.add(src_idx)
        has_prev.add(tgt_idx)

    # Find short orphans (no prev, no next, tokens < threshold)
    orphan_indices = set()
    for idx, entry in enumerate(entries):
        if idx not in has_next and idx not in has_prev:
            if entry["tokens"] < merge_short_threshold:
                orphan_indices.add(idx)

    if not orphan_indices:
        logger.info("No short orphans to merge")
        return entries

    logger.info(
        "Found %d short orphan entries (threshold %d tokens)",
        len(orphan_indices), merge_short_threshold,
    )

    # Group orphans by source, preserving original order
    source_orphans = defaultdict(list)
    for idx in sorted(orphan_indices):
        source_orphans[entries[idx]["source"]].append(idx)

    # Greedy merge within each source
    merged_entries = []  # new merged entries to add
    removed_indices = set()  # original indices consumed by merges

    for source, indices in source_orphans.items():
        current_group = []
        current_tokens = 0

        for idx in indices:
            entry = entries[idx]
            tok = entry["tokens"]

            if current_group and current_tokens + tok > merge_max_tokens:
                # Flush current group
                if len(current_group) >= 2:
                    merged = _make_merged_entry(entries, current_group, source)
                    merged_entries.append(merged)
                    removed_indices.update(current_group)
                current_group = [idx]
                current_tokens = tok
            else:
                current_group.append(idx)
                current_tokens += tok

        # Flush remaining
        if len(current_group) >= 2:
            merged = _make_merged_entry(entries, current_group, source)
            merged_entries.append(merged)
            removed_indices.update(current_group)

    if not removed_indices:
        logger.info("No orphan groups large enough to merge (need >= 2)")
        return entries

    # Rebuild entries: keep non-removed, append merged
    new_entries = [e for idx, e in enumerate(entries) if idx not in removed_indices]
    new_entries.extend(merged_entries)

    n_merged = len(merged_entries)
    n_removed = len(removed_indices)
    logger.info(
        "Merged %d orphan entries into %d merged entries (net -%d)",
        n_removed, n_merged, n_removed - n_merged,
    )

    return new_entries


def _make_merged_entry(entries: list, indices: list, source: str) -> dict:
    """Create a merged index entry from a group of orphan indices."""
    parts = []
    total_tokens = 0
    for idx in indices:
        e = entries[idx]
        parts.append({"file": e["file"], "offset": e["offset"]})
        total_tokens += e["tokens"]

    return {
        "source": source,
        "id": f"merged_{indices[0]}",
        "split": 0,
        "tokens": total_tokens,
        "parts": parts,
    }


def split_by_document(
    entries: list,
    val_ratio: float,
    test_ratio: float = 0.0,
    seed: int = 42,
):
    """Split entries into train/val/test at the document level (stratified by source).

    Groups entries by (source, id). For each source, randomly holds out
    val_ratio fraction of documents for validation and test_ratio fraction
    for testing. All chunks of the same document go to the same split,
    preserving continuation pairs.

    Args:
        entries: list of index entries.
        val_ratio: fraction of documents per source to hold out for val.
        test_ratio: fraction of documents per source to hold out for test.
        seed: random seed for reproducible splits.

    Returns:
        (train_entries, val_entries, test_entries) tuple.
        test_entries is always a list (empty when test_ratio == 0).
    """
    # Group entry indices by (source, id)
    doc_entries = defaultdict(list)
    for idx, entry in enumerate(entries):
        key = (entry["source"], entry["id"])
        doc_entries[key].append(idx)

    # Group document keys by source for stratified splitting
    source_docs = defaultdict(list)
    for key in doc_entries:
        source_docs[key[0]].append(key)

    rng = random.Random(seed)
    val_doc_keys = set()
    test_doc_keys = set()

    for source, doc_keys in sorted(source_docs.items()):
        rng.shuffle(doc_keys)
        n_docs = len(doc_keys)
        n_val = max(1, int(n_docs * val_ratio)) if val_ratio > 0 else 0
        n_test = max(1, int(n_docs * test_ratio)) if test_ratio > 0 else 0
        # Clamp so val + test never exceeds total docs
        n_val = min(n_val, n_docs)
        n_test = min(n_test, max(0, n_docs - n_val))

        val_keys = doc_keys[:n_val]
        test_keys = doc_keys[n_val:n_val + n_test]
        val_doc_keys.update(val_keys)
        test_doc_keys.update(test_keys)

        log_parts = [f"{n_val} val ({100.0*n_val/n_docs:.2f}%)"]
        if test_ratio > 0:
            log_parts.append(f"{n_test} test ({100.0*n_test/n_docs:.2f}%)")
        logger.info(
            "  %-15s  %5d docs total, %s",
            source, n_docs, ", ".join(log_parts),
        )

    # Partition entries
    train_entries = []
    val_entries = []
    test_entries = []
    for idx, entry in enumerate(entries):
        key = (entry["source"], entry["id"])
        if key in val_doc_keys:
            val_entries.append(entry)
        elif key in test_doc_keys:
            test_entries.append(entry)
        else:
            train_entries.append(entry)

    return train_entries, val_entries, test_entries


def _log_split_stats(entries: list, continuation_pairs: list, label: str):
    """Log per-source stats for a split."""
    source_counts = defaultdict(int)
    source_tokens = defaultdict(int)
    n_merged = 0
    for entry in entries:
        source_counts[entry["source"]] += 1
        source_tokens[entry["source"]] += entry["tokens"]
        if "parts" in entry:
            n_merged += 1
    for source in sorted(source_counts):
        logger.info(
            "  [%s] %-15s  %7d entries  %10d tokens",
            label, source, source_counts[source], source_tokens[source],
        )
    if n_merged:
        logger.info("  [%s] Merged entries: %d", label, n_merged)

    has_prev = set(p[1] for p in continuation_pairs)
    n_has_prev = len(has_prev)
    n_total = len(entries)
    logger.info(
        "  [%s] %d entries (%d has-prev recon/cont, %d no-prev NTP/recon), "
        "%d continuation pairs",
        label, n_total, n_has_prev, n_total - n_has_prev,
        len(continuation_pairs),
    )


def _write_index(entries: list, continuation_pairs: list, output_path: str, label: str):
    """Write an index file."""
    index = {
        "entries": entries,
        "continuation_pairs": continuation_pairs,
        "total_entries": len(entries),
        "total_continuation_pairs": len(continuation_pairs),
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(index, f)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info("[%s] Index written to %s (%.1f MB)", label, output_path, file_size_mb)


def main():
    parser = argparse.ArgumentParser(
        description="Build byte-offset index for formatted JSONL files."
    )
    parser.add_argument(
        "--data_dirs",
        nargs="+",
        required=True,
        help="Directories containing formatted JSONL files.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/train_index.json",
        help="Output path for the train index JSON file.",
    )
    parser.add_argument(
        "--upsample",
        nargs="*",
        default=[],
        help="Upsample sources, e.g. --upsample moegirl:3 games:2",
    )
    parser.add_argument(
        "--merge_max_tokens",
        type=int,
        default=3500,
        help="Max combined tokens for merged short orphan chunks.",
    )
    parser.add_argument(
        "--merge_short_threshold",
        type=int,
        default=2048,
        help="Only merge orphan chunks below this token count.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.001,
        help="Fraction of documents per source to hold out for validation. "
        "Set to 0.0 to disable val split. Default: 0.001 (0.1%%).",
    )
    parser.add_argument(
        "--val_output_path",
        type=str,
        default=None,
        help="Output path for the val index JSON file. "
        "Default: {output_path stem}_val.json.",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.0,
        help="Fraction of documents per source to hold out for testing. "
        "Set to 0.0 to disable test split (default). E.g. 0.009 (0.9%%).",
    )
    parser.add_argument(
        "--test_output_path",
        type=str,
        default=None,
        help="Output path for the test index JSON file. "
        "Default: {output_path stem}_test.json.",
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="Random seed for document split. Default: 42.",
    )
    args = parser.parse_args()

    if args.val_ratio < 0 or args.val_ratio >= 1.0:
        parser.error("--val_ratio must be in [0.0, 1.0)")
    if args.test_ratio < 0 or args.test_ratio >= 1.0:
        parser.error("--test_ratio must be in [0.0, 1.0)")
    if args.val_ratio + args.test_ratio >= 1.0:
        parser.error("--val_ratio + --test_ratio must be < 1.0")

    if args.merge_max_tokens < args.merge_short_threshold:
        logger.warning(
            "merge_max_tokens (%d) < merge_short_threshold (%d): "
            "orphans between these values will never merge.",
            args.merge_max_tokens, args.merge_short_threshold,
        )

    # Default output paths
    stem, ext = os.path.splitext(args.output_path)
    if args.val_output_path is None and args.val_ratio > 0:
        args.val_output_path = f"{stem}_val{ext}"
    if args.test_output_path is None and args.test_ratio > 0:
        args.test_output_path = f"{stem}_test{ext}"

    # --- Step 1: Scan all JSONL files ---
    all_entries = []
    for data_dir in args.data_dirs:
        if not os.path.isdir(data_dir):
            logger.warning("Directory not found, skipping: %s", data_dir)
            continue
        jsonl_files = sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".jsonl")
        )
        logger.info(
            "Scanning %d JSONL files in %s", len(jsonl_files), data_dir
        )
        for filepath in jsonl_files:
            entries = scan_jsonl_file(filepath)
            all_entries.extend(entries)
            logger.info(
                "  %s: %d entries", os.path.basename(filepath), len(entries)
            )

    logger.info("Total entries after scan: %d", len(all_entries))

    # --- Step 2: Train/val/test split (before upsample/merge) ---
    if args.val_ratio > 0 or args.test_ratio > 0:
        logger.info(
            "Splitting by document: val_ratio=%.4f, test_ratio=%.4f, seed=%d",
            args.val_ratio, args.test_ratio, args.split_seed,
        )
        train_entries, val_entries, test_entries = split_by_document(
            all_entries,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.split_seed,
        )
        logger.info(
            "Split result: %d train, %d val, %d test entries",
            len(train_entries), len(val_entries), len(test_entries),
        )
        val_entries = val_entries if args.val_ratio > 0 else None
        test_entries = test_entries if args.test_ratio > 0 else None
    else:
        train_entries = all_entries
        val_entries = None
        test_entries = None

    # --- Step 3: Process train split ---
    logger.info("=== Processing train split ===")

    # Upsample (train only)
    train_entries = upsample_entries(train_entries, args.upsample)
    if args.upsample:
        logger.info("[train] Total entries after upsample: %d", len(train_entries))

    # Build continuation pairs
    train_pairs = build_continuation_pairs(train_entries)
    logger.info("[train] Continuation pairs: %d", len(train_pairs))

    # Merge short orphans
    train_entries = merge_short_orphans(
        train_entries,
        train_pairs,
        merge_max_tokens=args.merge_max_tokens,
        merge_short_threshold=args.merge_short_threshold,
    )
    logger.info("[train] Total entries after merge: %d", len(train_entries))

    # Rebuild continuation pairs (indices shifted after merge)
    train_pairs = build_continuation_pairs(train_entries)
    logger.info("[train] Final continuation pairs: %d", len(train_pairs))

    _log_split_stats(train_entries, train_pairs, "train")
    _write_index(train_entries, train_pairs, args.output_path, "train")

    # --- Step 4: Process val split (no upsample, no merge) ---
    if val_entries is not None:
        logger.info("=== Processing val split ===")

        val_pairs = build_continuation_pairs(val_entries)
        logger.info("[val] Continuation pairs: %d", len(val_pairs))

        _log_split_stats(val_entries, val_pairs, "val")
        _write_index(val_entries, val_pairs, args.val_output_path, "val")

    # --- Step 5: Process test split (no upsample, no merge) ---
    if test_entries is not None:
        logger.info("=== Processing test split ===")

        test_pairs = build_continuation_pairs(test_entries)
        logger.info("[test] Continuation pairs: %d", len(test_pairs))

        _log_split_stats(test_entries, test_pairs, "test")
        _write_index(test_entries, test_pairs, args.test_output_path, "test")


if __name__ == "__main__":
    main()

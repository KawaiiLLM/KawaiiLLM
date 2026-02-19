"""Standalone script to build a byte-offset index over formatted JSONL files.

Scans all JSONL files in the given directories, records byte offset per line,
identifies continuation pairs, optionally upsamples small sources, and merges
short orphan chunks.

Usage:
    python src/train/build_index.py \
        --data_dirs data/novels/formatted data/bilibili/formatted ... \
        --output_path data/train_index.json \
        --upsample moegirl:3 \
        --merge_max_tokens 3500 --merge_short_threshold 2048
"""

import argparse
import json
import logging
import os
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
        help="Output path for the index JSON file.",
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
    args = parser.parse_args()

    if args.merge_max_tokens < args.merge_short_threshold:
        logger.warning(
            "merge_max_tokens (%d) < merge_short_threshold (%d): "
            "orphans between these values will never merge.",
            args.merge_max_tokens, args.merge_short_threshold,
        )

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

    # --- Step 2: Upsample specified sources ---
    all_entries = upsample_entries(all_entries, args.upsample)
    if args.upsample:
        logger.info("Total entries after upsample: %d", len(all_entries))

    # --- Step 3: Build continuation pairs ---
    continuation_pairs = build_continuation_pairs(all_entries)
    logger.info("Total continuation pairs: %d", len(continuation_pairs))

    # --- Step 4: Merge short orphan chunks ---
    all_entries = merge_short_orphans(
        all_entries,
        continuation_pairs,
        merge_max_tokens=args.merge_max_tokens,
        merge_short_threshold=args.merge_short_threshold,
    )
    logger.info("Total entries after merge: %d", len(all_entries))

    # Rebuild continuation pairs on final entries (indices may have shifted)
    continuation_pairs = build_continuation_pairs(all_entries)
    logger.info("Final continuation pairs: %d", len(continuation_pairs))

    # --- Step 5: Log per-source stats ---
    source_counts = defaultdict(int)
    source_tokens = defaultdict(int)
    n_merged = 0
    for entry in all_entries:
        source_counts[entry["source"]] += 1
        source_tokens[entry["source"]] += entry["tokens"]
        if "parts" in entry:
            n_merged += 1
    for source in sorted(source_counts):
        logger.info(
            "  %-15s  %7d entries  %10d tokens",
            source, source_counts[source], source_tokens[source],
        )
    logger.info("  Merged entries: %d", n_merged)

    # Has-next stats (for 2-task vs 3-task rotation)
    has_next = set(p[0] for p in continuation_pairs)
    n_has_next = len(has_next)
    n_total = len(all_entries)
    logger.info(
        "Task rotation: %d/%d (%.1f%%) entries have next chunk (3-task), "
        "%d (%.1f%%) entries without (2-task)",
        n_has_next, n_total, 100.0 * n_has_next / n_total if n_total else 0,
        n_total - n_has_next,
        100.0 * (n_total - n_has_next) / n_total if n_total else 0,
    )

    # --- Step 6: Write index ---
    index = {
        "entries": all_entries,
        "continuation_pairs": continuation_pairs,
        "total_entries": len(all_entries),
        "total_continuation_pairs": len(continuation_pairs),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(index, f)

    file_size_mb = os.path.getsize(args.output_path) / (1024 * 1024)
    logger.info("Index written to %s (%.1f MB)", args.output_path, file_size_mb)


if __name__ == "__main__":
    main()

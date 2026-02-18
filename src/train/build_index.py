"""Standalone script to build a byte-offset index over formatted JSONL files.

Scans all JSONL files in the given directories, records byte offset per line,
and identifies continuation pairs (consecutive splits of the same document).

Usage:
    python src/train/build_index.py \
        --data_dirs data/novels/formatted data/bilibili/formatted ... \
        --output_path data/train_index.json
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
    source = os.path.basename(os.path.dirname(filepath))

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
                "source": source,
                "id": record.get("id", ""),
                "split": record.get("split", 0),
                "tokens": record.get("token_count", 0),
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
    args = parser.parse_args()

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

    logger.info("Total entries: %d", len(all_entries))

    continuation_pairs = build_continuation_pairs(all_entries)
    logger.info("Total continuation pairs: %d", len(continuation_pairs))

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

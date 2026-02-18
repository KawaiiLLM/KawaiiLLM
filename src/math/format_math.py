import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow.parquet as pq
from transformers import AutoTokenizer
from tqdm import tqdm

from utils.chunking import chunk_by_tokens

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Format Math Corpus data into unified JSONL for pretraining.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing Parquet files (e.g. data/math)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to output JSONL file (e.g. data/math/formatted/math_formatted.jsonl)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="/Users/zhaoqixuan/Projects/models/Qwen3-0.6B",
        help="Local tokenizer directory for computing token lengths (default: Qwen3-0.6B).",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Maximum number of tokens per chunk.",
    )
    return parser.parse_args()


def iter_parquet_files(root: str):
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Input dir not found: {root}")

    for path in root_path.rglob("*.parquet"):
        yield path


def main():
    args = parse_args()

    logger.info("Loading tokenizer from local dir: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, local_files_only=True,
    )

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = list(iter_parquet_files(args.input_dir))
    logger.info("Found %d parquet files under %s", len(files), args.input_dir)

    total_chunks = 0
    with out_path.open("w", encoding="utf-8") as out_f:
        for path in tqdm(files, desc="Processing parquet files"):
            try:
                pf = pq.ParquetFile(path)
                total_rows = pf.metadata.num_rows
                with tqdm(total=total_rows, desc=path.name, leave=False, unit="rows") as pbar:
                    for batch in pf.iter_batches(batch_size=2048, columns=["uid", "content"]):
                        col = batch.to_pydict()
                        contents = col["content"]
                        uids     = col["uid"]

                        for content, uid in zip(contents, uids):
                            pbar.update(1)
                            if not isinstance(content, str) or not content.strip():
                                continue
                            if len(content) < 50:
                                continue

                            doc_id = uid or hashlib.md5(content.encode("utf-8")).hexdigest()

                            chunks = chunk_by_tokens(content, tokenizer, args.max_tokens)
                            if not chunks:
                                continue

                            for idx, chunk in enumerate(chunks):
                                record = {
                                    "source": "math",
                                    "id": str(doc_id),
                                    "split": idx,
                                    "tokens": len(tokenizer.encode(chunk, add_special_tokens=False)),
                                    "text": chunk,
                                }
                                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                                total_chunks += 1

            except Exception as e:
                logger.error("Error processing file %s: %s", path, e)

    logger.info("Done. Wrote %d chunks to %s", total_chunks, out_path)


if __name__ == "__main__":
    main()

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
        description="Format MoeGirl data into unified JSONL for pretraining.",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to input JSONL file (e.g. data/moegirl/cleaned/MoeGirlPedia_zh_cleaned_latest.jsonl)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to output JSONL file (e.g. data/moegirl/formatted/moegirl_formatted.jsonl)",
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


def main():
    args = parse_args()

    logger.info("Loading tokenizer from local dir: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, local_files_only=True,
    )

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    input_file = Path(args.input_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    total_chunks = 0
    with out_path.open("w", encoding="utf-8") as out_f:
        with input_file.open("r", encoding="utf-8") as in_f:
            pbar = tqdm(in_f, desc=f"Processing {input_file.name}", unit=" lines")
            for line_idx, line in enumerate(pbar):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse JSON at line %d", line_idx + 1)
                    continue

                title = data.get("title", "").strip()
                text = data.get("text", "").strip()

                if not text:
                    continue

                if len(text) < 50:
                    continue

                # Ensure title is at the beginning as a standalone line
                first_line = text.split('\n', 1)[0].strip()
                if title and first_line != title:
                    text = f"{title}\n\n{text}"

                chunks = chunk_by_tokens(text, tokenizer, args.max_tokens)
                if not chunks:
                    continue

                article_id = title if title else f"moegirl_{line_idx}"

                for idx, chunk in enumerate(chunks):
                    record = {
                        "source": "moegirl",
                        "id": str(article_id),
                        "split": idx,
                        "tokens": len(tokenizer.encode(chunk, add_special_tokens=False)),
                        "text": chunk,
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_chunks += 1

                if line_idx % 1000 == 0:
                    pbar.set_postfix({"chunks": total_chunks})

    logger.info("Done. Wrote %d chunks to %s", total_chunks, out_path)


if __name__ == "__main__":
    main()

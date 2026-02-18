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
        description="Format Code Corpus data into unified JSONL for pretraining.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Root directory containing code data (e.g. data/code)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save formatted JSONL files (e.g. data/code/formatted)",
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


def iter_jsonl_files(root: str):
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Input dir not found: {root}")

    for lang_dir in root_path.iterdir():
        if lang_dir.is_dir() and lang_dir.name != "formatted":
            for path in lang_dir.glob("*.jsonl"):
                yield path, lang_dir.name


def main():
    args = parse_args()

    logger.info("Loading tokenizer from local dir: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, local_files_only=True,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for file_path, lang in iter_jsonl_files(args.input_dir):
        output_file = output_dir / f"code_{lang}_formatted.jsonl"
        logger.info("Processing %s -> %s", file_path, output_file)

        total_chunks = 0
        with output_file.open("w", encoding="utf-8") as out_f:
            with file_path.open("r", encoding="utf-8") as in_f:
                for line in tqdm(in_f, desc=f"Processing {lang}"):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    content = data.get("content", "")
                    if not content or len(content) < 50:
                        continue

                    meta = {
                        "repo_name": data.get("max_stars_repo_name"),
                        "path": data.get("max_stars_repo_path"),
                        "stars": data.get("max_stars_count"),
                        "lang": lang
                    }

                    original_id = data.get("id", "")
                    doc_id = f"{lang}_{original_id}"

                    # 代码模式：跳过标点切分
                    chunks = chunk_by_tokens(
                        content, tokenizer, args.max_tokens,
                        skip_sentence_split=True,
                    )

                    for idx, chunk in enumerate(chunks):
                        record = {
                            "source": "code",
                            "id": doc_id,
                            "split": idx,
                            "tokens": len(tokenizer.encode(chunk, add_special_tokens=False)),
                            "text": chunk,
                            "meta": meta
                        }
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        total_chunks += 1

        logger.info("Finished %s: %d chunks", lang, total_chunks)

    logger.info("All done.")


if __name__ == "__main__":
    main()

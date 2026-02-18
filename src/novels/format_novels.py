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
        description="Format light novels data into unified JSONL for pretraining.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Root directory containing deduped novel JSON files (e.g. data/novels/deduped)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to output JSONL file (e.g. data/novels/formatted/novels_formatted.jsonl)",
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


def iter_json_files(root: str):
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Input dir not found: {root}")

    for path in root_path.rglob("*.json"):
        yield path


def load_novel(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to load %s: %s", path, e)
        return None


def concatenate_text(data: dict) -> str:
    texts = data.get("texts")
    if not isinstance(texts, dict):
        return ""

    # keys are string indices: "0", "1", ...
    pieces = []
    for k in sorted(texts.keys(), key=lambda x: int(x)):
        item = texts.get(k) or {}
        t = item.get("text")
        if isinstance(t, str):
            pieces.append(t)
    return "\n\n".join(pieces)


def main():
    args = parse_args()

    logger.info("Loading tokenizer from local dir: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, local_files_only=True,
    )

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = list(iter_json_files(args.input_dir))
    logger.info("Found %d novel files under %s", len(files), args.input_dir)

    total_chunks = 0
    with out_path.open("w", encoding="utf-8") as out_f:
        for path in tqdm(files, desc="Processing novels"):
            data = load_novel(path)
            if not data:
                continue

            full_text = concatenate_text(data)
            if not full_text.strip():
                continue

            # Check total character count before chunking
            if len(full_text) < 4000:
                continue

            meta = data.get("meta", {})
            novel_id = meta.get("id") or path.stem
            title = meta.get("title", "")

            # C2 修复：标题在 chunking 之前拼入，确保不超 max_tokens
            if title:
                full_text = f"{title}\n\n{full_text}"

            chunks = chunk_by_tokens(full_text, tokenizer, args.max_tokens)
            if not chunks:
                continue

            for idx, chunk in enumerate(chunks):
                record = {
                    "source": "novels",
                    "id": str(novel_id),
                    "split": idx,
                    "tokens": len(tokenizer.encode(chunk, add_special_tokens=False)),
                    "text": chunk,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_chunks += 1

    logger.info("Done. Wrote %d chunks to %s", total_chunks, out_path)


if __name__ == "__main__":
    main()

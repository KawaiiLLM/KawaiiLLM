import argparse
import json
import logging
import re
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
        description="Format Game Scripts data into unified JSONL for pretraining.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Root directory containing game script text files (e.g. data/games/raw/zh)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to output JSONL file (e.g. data/games/formatted/games_formatted.jsonl)",
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


def iter_text_files(root: str):
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Input dir not found: {root}")

    for path in root_path.rglob("*.txt"):
        yield path


def read_file_content(path: Path) -> str:
    """Try reading file with utf-8, fallback to gb18030 then latin-1."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with path.open("r", encoding="gb18030") as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with path.open("r", encoding="latin-1") as f:
                    return f.read()
            except Exception as e:
                logger.warning("Failed to read %s: %s", path, e)
                return ""
        except Exception as e:
            logger.warning("Failed to read %s: %s", path, e)
            return ""
    except Exception as e:
        logger.warning("Failed to read %s: %s", path, e)
        return ""


def main():
    args = parse_args()

    logger.info("Loading tokenizer from local dir: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, local_files_only=True,
    )

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = list(iter_text_files(args.input_dir))
    logger.info("Found %d script files under %s", len(files), args.input_dir)

    total_chunks = 0
    with out_path.open("w", encoding="utf-8") as out_f:
        for path in tqdm(files, desc="Processing scripts"):
            text = read_file_content(path)
            if not text or not text.strip():
                continue

            if len(text) < 1000:
                continue

            # Extract metadata from path
            try:
                rel_path = path.relative_to(args.input_dir)
                parts = rel_path.parts

                if len(parts) >= 2:
                    company = parts[0]
                    game = parts[1]
                    game = game.split('@')[0]

                    chapter = path.stem
                    chapter = chapter.split('@')[0]
                    chapter = re.sub(r'(_zh|_cn|_f)+$', '', chapter, flags=re.IGNORECASE)

                    title = f"{game} - {chapter}"
                    script_id = f"{company}_{game}_{chapter}"
                    script_id = re.sub(r'[^\w\-_]', '_', script_id)
                else:
                    script_id = path.stem
                    title = script_id.replace("_", " ")
            except Exception:
                title = path.stem
                script_id = path.stem

            # Prepend title before chunking
            text = f"{title}\n\n{text}"

            chunks = chunk_by_tokens(text, tokenizer, args.max_tokens)
            if not chunks:
                continue

            for idx, chunk in enumerate(chunks):
                record = {
                    "source": "games",
                    "id": str(script_id),
                    "split": idx,
                    "tokens": len(tokenizer.encode(chunk, add_special_tokens=False)),
                    "text": chunk,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_chunks += 1

    logger.info("Done. Wrote %d chunks to %s", total_chunks, out_path)


if __name__ == "__main__":
    main()

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
        description="Format Bilibili articles data into unified JSONL for pretraining.",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input JSONL file or directory containing JSONL files (e.g. data/bilibili/cleaned)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to output JSONL file (e.g. data/bilibili/formatted/bilibili_formatted.jsonl)",
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


def iter_input_files(path_str: str):
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {path_str}")

    if path.is_file():
        yield path
    elif path.is_dir():
        for p in path.rglob("*.jsonl"):
            yield p
    else:
        raise ValueError(f"Invalid input path: {path_str}")


def format_article(data: dict) -> str:
    """Format article with metadata and filtered comments."""
    title = data.get("title", "").strip()
    content = data.get("content", "").strip()
    if not content:
        return ""

    # 1. Build Metadata Header
    meta_parts = []

    publish_time = data.get("publish_time")
    if publish_time:
        meta_parts.append(publish_time)

    author = data.get("author_name")
    if author:
        meta_parts.append(author)

    tags = data.get("tags")
    if tags:
        if isinstance(tags, str):
            try:
                if tags.startswith("["):
                    tags_list = json.loads(tags)
                    if isinstance(tags_list, list):
                        tags_str = ",".join(str(t) for t in tags_list)
                        meta_parts.append(tags_str)
                    else:
                        meta_parts.append(tags)
                else:
                    meta_parts.append(tags)
            except Exception:
                meta_parts.append(tags)
        elif isinstance(tags, list):
             meta_parts.append(",".join(str(t) for t in tags))

    # Format: Title \n\n Time | Author | Tags \n\n Content
    meta_header = f"{title}\n\n"
    if meta_parts:
        meta_header += " | ".join(meta_parts) + "\n\n"
    else:
        meta_header += "\n"

    # 2. Filter and Format Comments
    comments = data.get("comments", [])
    valid_root_comments = []

    if isinstance(comments, list):
        for c in comments:
            root_content = c.get("content", "").strip()
            root_author = c.get("author_name", "Unknown")
            root_like = c.get("like", 0)
            root_id = c.get("rpid")

            if len(root_content) >= 10:
                replies = c.get("replies", [])
                valid_replies = []
                if isinstance(replies, list):
                    for r in replies:
                        if r.get("parent") != root_id:
                            continue

                        reply_content = r.get("content", "").strip()
                        match = re.match(r"^回复 @.+? :", reply_content)

                        effective_len = len(reply_content)
                        if match:
                            effective_len -= len(match.group(0))

                        if effective_len >= 10:
                            if match:
                                reply_content = reply_content[len(match.group(0)):]

                            valid_replies.append({
                                "content": reply_content,
                                "like": r.get("like", 0),
                                "author": r.get("author_name", "Unknown")
                            })

                valid_replies.sort(key=lambda x: x["like"], reverse=True)
                top_replies = valid_replies[:2]

                valid_root_comments.append({
                    "content": root_content,
                    "like": root_like,
                    "author": root_author,
                    "replies": top_replies
                })

    comments_section = ""
    if valid_root_comments:
        valid_root_comments.sort(key=lambda x: x["like"], reverse=True)
        top_root_comments = valid_root_comments[:5]

        comments_section = "\n\n---\n评论：\n"
        for c in top_root_comments:
            clean_root_content = c["content"].replace("\n", " ")
            comments_section += f"{c['author']}: {clean_root_content}\n"

            for r in c["replies"]:
                clean_reply_content = r["content"].replace("\n", " ")
                comments_section += f"  └─ {r['author']}: {clean_reply_content}\n"

    return meta_header + content + comments_section


def main():
    args = parse_args()

    logger.info("Loading tokenizer from local dir: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, local_files_only=True,
    )

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    input_files = list(iter_input_files(args.input_path))
    logger.info("Found %d input files under %s", len(input_files), args.input_path)

    total_chunks = 0
    with out_path.open("w", encoding="utf-8") as out_f:
        for path in tqdm(input_files, desc="Processing files"):
            try:
                with path.open("r", encoding="utf-8") as in_f:
                    pbar = tqdm(in_f, desc=f"Processing {path.name}", unit=" lines")
                    for line_idx, line in enumerate(pbar):
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            logger.warning("Failed to parse JSON at %s:%d", path, line_idx + 1)
                            continue

                        full_text = format_article(data)
                        if not full_text.strip():
                            continue

                        chunks = chunk_by_tokens(full_text, tokenizer, args.max_tokens)
                        if not chunks:
                            continue

                        article_id = data.get("cvid")
                        if article_id is None:
                            article_id = data.get("id") or f"{path.stem}_{line_idx}"

                        for idx, chunk in enumerate(chunks):
                            record = {
                                "source": "bilibili",
                                "id": str(article_id),
                                "split": idx,
                                "tokens": len(tokenizer.encode(chunk, add_special_tokens=False)),
                                "text": chunk,
                            }
                            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                            total_chunks += 1

                        if line_idx % 100 == 0:
                            pbar.set_postfix({"chunks": total_chunks})

            except Exception as e:
                logger.error("Error processing file %s: %s", path, e)

    logger.info("Done. Wrote %d chunks to %s", total_chunks, out_path)


if __name__ == "__main__":
    main()

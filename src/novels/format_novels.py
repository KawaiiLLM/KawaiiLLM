import argparse
import json
import logging
import os
import re
from pathlib import Path

from transformers import AutoTokenizer
from tqdm import tqdm

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


def split_by_sentence(text: str) -> list:
    """按结束标点切分，标点保留在前一段末尾。

    支持中文标点（。！？…；）和英文标点（! ? ;）以及换行符。
    """
    parts = re.split(r'(?<=[。！？…；!?\n])', text)
    return [p for p in parts if p]


def hard_split_by_chars(text: str, tokenizer, max_tokens: int) -> list:
    """字符级兜底硬切：用二分查找找到不超过 max_tokens 的最长前缀。

    保证不破坏字符完整性。
    """
    blocks = []
    remaining = text
    while remaining:
        tok_len = len(tokenizer.encode(remaining, add_special_tokens=False))
        if tok_len <= max_tokens:
            blocks.append(remaining)
            break

        # 二分查找：找到最大的 mid 使得 remaining[:mid] 的 token 数 <= max_tokens
        lo, hi = 1, len(remaining)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if len(tokenizer.encode(remaining[:mid], add_special_tokens=False)) <= max_tokens:
                lo = mid
            else:
                hi = mid - 1
        blocks.append(remaining[:lo])
        remaining = remaining[lo:]

    return blocks


def chunk_by_tokens(text: str, tokenizer, max_tokens: int) -> list:
    """将完整小说文本按 token 数切分为多个 chunk。

    切分策略（逐级细分）:
        1. 先按 \\n\\n 切分为段落。
        2. 若段落超过 max_tokens，再按 \\n 切分。
        3. 若仍超过，再按结束标点切分。
        4. 若仍超过，按字符级二分查找硬切。
        5. 将上述小块逐个累积，一旦累积超过 max_tokens，
           就把之前累积的内容作为一个 chunk 输出，
           当前块成为下一个 chunk 的起始。
    """
    if not text:
        return []

    def count_tokens(s: str) -> int:
        return len(tokenizer.encode(s, add_special_tokens=False))

    # ========== 第一步：按 \n\n 切分 ==========
    raw_paragraphs = text.split('\n\n')

    # 重新附加分隔符，保持原文还原性
    paragraphs_with_sep = []
    for i, para in enumerate(raw_paragraphs):
        suffix = "\n\n" if i < len(raw_paragraphs) - 1 else ""
        paragraphs_with_sep.append(para + suffix)

    # ========== 第二步：超长段落逐级细分 ==========
    refined_blocks: list = []

    for para_text in paragraphs_with_sep:
        if count_tokens(para_text) <= max_tokens:
            refined_blocks.append(para_text)
            continue

        # --- 2a: 按 \n 切分 ---
        lines = para_text.split('\n')
        line_pieces = []
        for j, line in enumerate(lines):
            line_suffix = "\n" if j < len(lines) - 1 else ""
            line_pieces.append(line + line_suffix)

        for line_text in line_pieces:
            if count_tokens(line_text) <= max_tokens:
                refined_blocks.append(line_text)
                continue

            # --- 2b: 按结束标点切分 ---
            sentences = split_by_sentence(line_text)
            for sent in sentences:
                if count_tokens(sent) <= max_tokens:
                    refined_blocks.append(sent)
                else:
                    # --- 2c: 字符级兜底硬切 ---
                    hard_blocks = hard_split_by_chars(sent, tokenizer, max_tokens)
                    refined_blocks.extend(hard_blocks)

    # ========== 第三步：累积合并为 chunk ==========
    chunks: list = []
    current_text = ""
    current_len = 0

    for block in refined_blocks:
        block_len = count_tokens(block)

        if current_len + block_len > max_tokens:
            # flush 前面累积的内容
            if current_text:
                chunks.append(current_text)
            # 新 block 作为新 chunk 的起点
            current_text = block
            current_len = block_len
        else:
            current_text += block
            current_len += block_len

    # flush 残余
    if current_text:
        chunks.append(current_text)

    return chunks


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

            chunks = chunk_by_tokens(full_text, tokenizer, args.max_tokens)
            if not chunks:
                continue

            meta = data.get("meta", {})
            novel_id = meta.get("id") or path.stem

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

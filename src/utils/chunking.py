"""共享的分层语义切分模块。

提供统一的文本切分逻辑，所有格式化脚本共享调用。

切分策略（逐级细分）:
    1. 按 \\n\\n 切分为段落。
    2. 若段落超过 max_tokens，按 \\n 切分。
    3. 若仍超过，按结束标点切分。
    4. 若仍超过，按字符级二分查找硬切。
    5. 将小块逐个累积，超过 max_tokens 时 flush。

代码模式（skip_sentence_split=True）:
    跳过第 3 步（标点切分），直接从行级切分跳到硬切。
"""

import re


# 统一的标点切分正则：中文标点 + 英文标点 + 换行
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[。！？…；!?;.\n])')


def split_by_sentence(text: str) -> list[str]:
    """按结束标点切分，标点保留在前一段末尾。

    支持中文标点（。！？…；）和英文标点（. ! ? ;）以及换行符。
    """
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p for p in parts if p]


def hard_split_by_chars(text: str, tokenizer, max_tokens: int) -> list[str]:
    """字符级兜底硬切：用二分查找找到不超过 max_tokens 的最长前缀。"""
    blocks: list[str] = []
    remaining = text
    while remaining:
        tok_len = len(tokenizer.encode(remaining, add_special_tokens=False))
        if tok_len <= max_tokens:
            blocks.append(remaining)
            break

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


def chunk_by_tokens(
    text: str,
    tokenizer,
    max_tokens: int,
    *,
    skip_sentence_split: bool = False,
) -> list[str]:
    """将完整文本按 token 数切分为多个 chunk。

    Args:
        text: 待切分文本。
        tokenizer: HuggingFace tokenizer 实例。
        max_tokens: 每个 chunk 的最大 token 数。
        skip_sentence_split: 为 True 时跳过标点切分（适用于代码）。

    Returns:
        切分后的 chunk 列表。
    """
    if not text:
        return []

    def count_tokens(s: str) -> int:
        return len(tokenizer.encode(s, add_special_tokens=False))

    # ========== 第一步：按 \n\n 切分 ==========
    raw_paragraphs = text.split('\n\n')

    paragraphs_with_sep: list[str] = []
    for i, para in enumerate(raw_paragraphs):
        suffix = "\n\n" if i < len(raw_paragraphs) - 1 else ""
        paragraphs_with_sep.append(para + suffix)

    # ========== 第二步：超长段落逐级细分 ==========
    refined_blocks: list[str] = []

    for para_text in paragraphs_with_sep:
        if count_tokens(para_text) <= max_tokens:
            refined_blocks.append(para_text)
            continue

        # --- 2a: 按 \n 切分 ---
        lines = para_text.split('\n')
        line_pieces: list[str] = []
        for j, line in enumerate(lines):
            line_suffix = "\n" if j < len(lines) - 1 else ""
            line_pieces.append(line + line_suffix)

        for line_text in line_pieces:
            if count_tokens(line_text) <= max_tokens:
                refined_blocks.append(line_text)
                continue

            if skip_sentence_split:
                # 代码模式：跳过标点切分，直接硬切
                refined_blocks.extend(
                    hard_split_by_chars(line_text, tokenizer, max_tokens)
                )
            else:
                # --- 2b: 按结束标点切分 ---
                sentences = split_by_sentence(line_text)
                for sent in sentences:
                    if count_tokens(sent) <= max_tokens:
                        refined_blocks.append(sent)
                    else:
                        # --- 2c: 字符级兜底硬切 ---
                        refined_blocks.extend(
                            hard_split_by_chars(sent, tokenizer, max_tokens)
                        )

    # ========== 第三步：累积合并为 chunk ==========
    # 修复 C1: 当累积 token 接近上限时，用实际 count_tokens 校验
    chunks: list[str] = []
    current_text = ""
    current_len = 0

    for block in refined_blocks:
        block_len = count_tokens(block)

        if current_len + block_len > max_tokens:
            if current_text:
                chunks.append(current_text)
            current_text = block
            current_len = block_len
        else:
            candidate = current_text + block
            # 当累积长度超过 max_tokens 的 90% 时，用精确计数校验
            estimated = current_len + block_len
            if estimated > max_tokens * 0.9:
                actual_len = count_tokens(candidate)
                if actual_len > max_tokens:
                    # 精确计数超限，flush 之前的内容
                    if current_text:
                        chunks.append(current_text)
                    current_text = block
                    current_len = block_len
                    continue
                current_len = actual_len
            else:
                current_len = estimated
            current_text = candidate

    if current_text:
        chunks.append(current_text)

    return chunks

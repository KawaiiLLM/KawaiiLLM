# Data Formatting Design

## Overview
This document outlines the design for formatting all datasets into a unified pre-training format suitable for the KawaiiLLM project, including: Novels, Bilibili, MoeGirl, Games, General, Math, Code.

## Goals
- Convert raw data from various formats (JSON/JSONL/TXT/Parquet) into unified JSONL files.
- Ensure each text chunk does not exceed **4096 tokens**.
- Use the **Qwen3-0.6B** tokenizer for accurate token counting.
- Implement **hierarchical semantic chunking** to preserve context integrity.
- Include rich metadata (Title, Time, Author, Tags, Comments) where applicable.

## 1. Shared Chunking Module (`src/utils/chunking.py`)

所有格式化脚本共享同一个分层切分模块，避免代码重复。

### 切分策略（Hierarchical Semantic Splitting）

1.  **段落切分**: 按 `\n\n` 切分。
2.  **行切分**: 若段落 > 4096 tokens，按 `\n` 切分。
3.  **句子切分**: 若行 > 4096 tokens，按结束标点切分（`。！？…；!?;.\n`）。
    - 代码模式 (`skip_sentence_split=True`) 跳过此步。
4.  **硬切**: 若句子 > 4096 tokens，用二分查找找到不超过 max_tokens 的最长字符前缀。

### 累积合并逻辑

- 遍历上述最小单元块，累积到 buffer。
- 若添加当前块后超过 max_tokens：flush buffer 为一个 chunk，当前块成为新 buffer 起点。
- **精确校验**：当累积估算长度 > 90% max_tokens 时，调用 `count_tokens()` 精确校验，防止 tokenization 不可加性导致的漂移。

### API

```python
from utils.chunking import chunk_by_tokens

# 普通文本切分
chunks = chunk_by_tokens(text, tokenizer, max_tokens=4096)

# 代码切分（跳过句子级切分）
chunks = chunk_by_tokens(code, tokenizer, max_tokens=4096, skip_sentence_split=True)
```

## 2. Novels Formatting (`src/novels/format_novels.py`)

### Input Format
- **Source**: `data/novels/deduped/**/*.json`
- **Structure**:
  ```json
  {
    "meta": { "id": "...", "title": "..." },
    "texts": { "0": {"text": "..."}, "1": {"text": "..."} }
  }
  ```

### Processing Logic
1.  **Concatenate**: Join all text segments with `\n\n`.
2.  **Filter**: Skip novels with total length < 4000 characters.
3.  **Prepend Title**: Add `{title}\n\n` to text **before chunking**（确保不超 max_tokens）。
4.  **Chunking**: Apply hierarchical semantic splitting.

### Output Format
- **File**: `data/novels/formatted/novels_formatted.jsonl`
- **Fields**: `source`, `id`, `split`, `tokens`, `text`

### 执行命令
```bash
.venv/bin/python src/novels/format_novels.py \
  --input_dir data/novels/deduped \
  --output_file data/novels/formatted/novels_formatted.jsonl
```

## 3. Bilibili Formatting (`src/bilibili/format_bilibili.py`)

### Input Format
- **Source**: `data/bilibili/cleaned/**/*.jsonl`
- **Structure**:
  ```json
  {
    "cvid": 123,
    "title": "...",
    "publish_time": "...",
    "author_name": "...",
    "tags": "...",
    "content": "...",
    "comments": [...]
  }
  ```

### Processing Logic
1.  **Metadata Formatting**: Construct a header string from metadata.
    ```text
    {title}

    {time} | {author} | {tags}

    {content}
    ```
2.  **Comment Filtering**:
    - **Root Comments**: Length >= 10 chars. Top 5 by likes.
    - **Replies**: Direct reply only, effective length >= 10 chars, remove `回复 @xxx :` prefix. Top 2 per root by likes.
    - **Format**:
      ```text
      ---
      评论：
      {root_author}: {root_content}
        └─ {reply_author}: {reply_content}
      ```
3.  **Concatenate**: `Header + Content + Comments`.
4.  **Chunking**: Apply hierarchical semantic splitting.

### Output Format
- **File**: `data/bilibili/formatted/bilibili_formatted.jsonl`
- **Fields**: `source`, `id`, `split`, `tokens`, `text`

### 执行命令
```bash
.venv/bin/python src/bilibili/format_bilibili.py \
  --input_path data/bilibili/cleaned \
  --output_file data/bilibili/formatted/bilibili_formatted.jsonl
```

## 4. MoeGirl Formatting (`src/moegirl/format_moegirl.py`)

### Input Format
- **Source**: `data/moegirl/cleaned/**/*.jsonl`
- **Structure**: `{ "title": "...", "text": "..." }`

### Processing Logic
1.  **Filter**: Skip articles with length < 50 characters.
2.  **Prepend Title**: If the first line of `text` != `title`, prepend `{title}\n\n`.
3.  **Chunking**: Apply hierarchical semantic splitting.

### Output Format
- **File**: `data/moegirl/formatted/moegirl_formatted.jsonl`
- **Fields**: `source`, `id`, `split`, `tokens`, `text`

### 执行命令
```bash
.venv/bin/python src/moegirl/format_moegirl.py \
  --input_file data/moegirl/cleaned/MoeGirlPedia_zh_cleaned_latest.jsonl \
  --output_file data/moegirl/formatted/moegirl_formatted.jsonl
```

## 5. Games Formatting (`src/games/format_games.py`)

### Input Format
- **Source**: `data/games/raw/zh/**/*.txt` and `data/games/raw/en/**/*.txt`
- **Structure**: Plain text files.

### Processing Logic
1.  **Encoding**: Try `utf-8` → `gb18030` → `latin-1`.
2.  **Filter**: Skip scripts with length < 1000 characters.
3.  **Metadata Extraction**: Company/Game from directory structure, Chapter from filename.
4.  **Prepend Title**: `{Game} - {Chapter}\n\n{Content}` (before chunking).
5.  **Chunking**: Apply hierarchical semantic splitting.

### Output Format
- **File**: `data/games/formatted/games_zh_formatted.jsonl` / `games_en_formatted.jsonl`
- **Fields**: `source`, `id`, `split`, `tokens`, `text`

### 执行命令
```bash
# 中文
.venv/bin/python src/games/format_games.py \
  --input_dir data/games/raw/zh \
  --output_file data/games/formatted/games_zh_formatted.jsonl

# 英文
.venv/bin/python src/games/format_games.py \
  --input_dir data/games/raw/en \
  --output_file data/games/formatted/games_en_formatted.jsonl
```

## 6. General Corpus Formatting (`src/general/format_general.py`)

### Input Format
- **Source**:
  - `data/general/Ultra-FineWeb-zh/*.parquet` (Chinese): columns `content`, `score`, `source`.
  - `data/general/ultrafineweb_en_v1_4/*.parquet` (English): columns `content`, `dataset_index`, `meta` (JSON), `score`, `source`, `uid`.

### Processing Logic
1.  **Loading**: Read Parquet files using `pandas`.
2.  **Filter**: Skip texts with length < 100 characters.
3.  **Metadata**: English: parse `meta` JSON for `url`/`date`. Chinese: no metadata.
4.  **ID**: English: `uid`. Chinese: `md5(content)`.
5.  **Chunking**: Apply hierarchical semantic splitting.

### Output Format
- **File**: `data/general/formatted/general_zh_formatted.jsonl` / `general_en_formatted.jsonl`
- **Fields**: `source`, `id`, `split`, `tokens`, `text`, `meta` (English only)

### 执行命令
```bash
# 中文
.venv/bin/python src/general/format_general.py \
  --input_dir data/general/Ultra-FineWeb-zh \
  --output_file data/general/formatted/general_zh_formatted.jsonl \
  --lang zh

# 英文
.venv/bin/python src/general/format_general.py \
  --input_dir data/general/ultrafineweb_en_v1_4 \
  --output_file data/general/formatted/general_en_formatted.jsonl \
  --lang en
```

## 7. Math Corpus Formatting (`src/math/format_math.py`)

### Input Format
- **Source**: `data/math/*.parquet`
- **Structure**: Parquet files with columns `uid`, `content`.

### Processing Logic
1.  **Loading**: Read Parquet files using `pandas`.
2.  **Filter**: Skip texts with length < 50 characters.
3.  **ID**: `uid`, fallback to `md5(content)`.
4.  **Chunking**: Apply hierarchical semantic splitting.

### Output Format
- **File**: `data/math/formatted/math_formatted.jsonl`
- **Fields**: `source`, `id`, `split`, `tokens`, `text`

### 执行命令
```bash
.venv/bin/python src/math/format_math.py \
  --input_dir data/math \
  --output_file data/math/formatted/math_formatted.jsonl
```

## 8. Code Corpus Formatting (`src/code/format_code.py`)

### Input Format
- **Source**: `data/code/{lang}/{lang}_stratified.jsonl`
- **Languages**: cpp, csharp, go, java, javascript, lua, python, rust, sql, typescript
- **Structure**: JSONL, fields: `content`, `max_stars_repo_name`, `max_stars_repo_path`, `max_stars_count`, `id`.

### Processing Logic
1.  **Loading**: Read JSONL files line by line.
2.  **Filter**: Skip texts with length < 50 characters.
3.  **Chunking**: 使用代码模式 (`skip_sentence_split=True`)：
    - 段落切分 (`\n\n`) → 行切分 (`\n`) → 硬切（无标点切分）。
4.  **Metadata**: Preserve repo info in `meta` field.

### Output Format
- **File**: `data/code/formatted/code_{lang}_formatted.jsonl`
- **Fields**: `source`, `id`, `split`, `tokens`, `text`, `meta`

### 执行命令
```bash
.venv/bin/python src/code/format_code.py \
  --input_dir data/code \
  --output_dir data/code/formatted
```

## 9. Data Merging & Shuffling (`src/merge_and_shuffle.py`)

### Strategy

#### Interleaved Reading
从多个文件中交错读取（每文件每次读 1000 行），避免同源数据扎堆。

#### Streaming Shuffle with Buffer
1. 交错读取所有输入文件的行到 buffer（默认 500,000 行）。
2. Buffer 满时 shuffle，flush 50% 到输出。
3. 重复直至所有输入耗尽。
4. Shuffle 并写入剩余 buffer。

#### Sharding
- 输出文件 `pretrain_v1_part_{idx:03d}.jsonl`，每个 shard 默认 100,000 行。
- 自动删除末尾空分片。

#### Validation Set
- 随机抽取 0.5% 行写入 `validation.jsonl`。

### Arguments
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input_dirs` | (required) | 输入目录列表 |
| `--output_dir` | (required) | 输出目录 |
| `--val_ratio` | 0.005 | 验证集比例 |
| `--shard_size` | 100000 | 每个分片的最大行数 |
| `--buffer_size` | 500000 | Shuffle buffer 大小 |
| `--seed` | 42 | 随机种子 |

### 执行命令
```bash
.venv/bin/python src/merge_and_shuffle.py \
  --input_dirs \
    data/novels/formatted \
    data/bilibili/formatted \
    data/moegirl/formatted \
    data/games/formatted \
    data/general/formatted \
    data/math/formatted \
    data/code/formatted \
  --output_dir data/pretrain \
  --shard_size 100000 \
  --buffer_size 500000
```

## 10. Implementation Details

### Dependencies
- `transformers`: For `AutoTokenizer`.
- `tqdm`: For progress bars.
- `pandas`, `pyarrow`: For reading Parquet files.

### Tokenizer
- **Model**: Local path to `Qwen3-0.6B`.
- **Loading**: `AutoTokenizer.from_pretrained(..., trust_remote_code=True, local_files_only=True)`.

### Error Handling
- Skip files with read errors.
- Skip empty content.
- Log warnings for malformed data.
- 文件句柄使用 context manager 确保异常时正确关闭。

### Project Structure
```
src/
├── utils/
│   └── chunking.py          # 共享分层切分模块
├── novels/
│   └── format_novels.py
├── bilibili/
│   └── format_bilibili.py
├── moegirl/
│   └── format_moegirl.py
├── games/
│   └── format_games.py
├── general/
│   └── format_general.py
├── math/
│   └── format_math.py
├── code/
│   └── format_code.py
└── merge_and_shuffle.py
```

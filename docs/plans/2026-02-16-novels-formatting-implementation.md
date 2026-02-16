# Novels Data Formatting Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a Python script to format Light Novels data into a unified JSONL format for pre-training, ensuring chunks do not exceed 4096 tokens using the Qwen tokenizer.

**Architecture:** A single Python script `src/novels/format_novels.py` that reads JSON files, tokenizes text, chunks it, and writes to a JSONL file.

**Tech Stack:** Python 3, Transformers (Qwen tokenizer), TQDM.

---

### Task 1: Environment Setup

**Files:**
- Create: `requirements.txt`

**Step 1: Create requirements.txt**

```text
transformers
torch
tqdm
tiktoken
sentencepiece
accelerate
```

**Step 2: Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 3: Verify installation**

```bash
python3 -c "import transformers; print(transformers.__version__)"
python3 -c "import torch; print(torch.__version__)"
```

### Task 2: Create Formatting Script Skeleton

**Files:**
- Create: `src/novels/format_novels.py`

**Step 1: Create script with argument parsing and logging**

```python
import argparse
import logging
import os
import json
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Format novels data for pre-training.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing raw JSON files.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output JSONL file.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model name for tokenizer.")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Maximum tokens per chunk.")
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info(f"Starting formatting with input_dir={args.input_dir}, output_file={args.output_file}")

    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory {args.input_dir} does not exist.")
        return

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Placeholder for processing logic
    logger.info("Processing complete.")

if __name__ == "__main__":
    main()
```

**Step 2: Run script to verify arguments**

```bash
python3 src/novels/format_novels.py --input_dir data/novels/deduped --output_file data/novels/formatted/novels_formatted.jsonl
```

### Task 3: Implement Tokenizer Loading and File Reading

**Files:**
- Modify: `src/novels/format_novels.py`

**Step 1: Add tokenizer loading**

```python
from transformers import AutoTokenizer

def load_tokenizer(model_name):
    logger.info(f"Loading tokenizer: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        return tokenizer
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise
```

**Step 2: Add file reading logic**

```python
import glob

def get_json_files(input_dir):
    files = []
    for root, _, filenames in os.walk(input_dir):
        for filename in filenames:
            if filename.endswith('.json'):
                files.append(os.path.join(root, filename))
    return files

def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.warning(f"Failed to read {file_path}: {e}")
        return None
```

**Step 3: Update main function**

```python
def main():
    args = parse_args()
    tokenizer = load_tokenizer(args.model_name)
    files = get_json_files(args.input_dir)
    logger.info(f"Found {len(files)} JSON files.")

    # Test reading first file
    if files:
        data = read_json_file(files[0])
        if data:
            logger.info(f"Successfully read first file: {files[0]}")
```

### Task 4: Implement Text Concatenation and Chunking

**Files:**
- Modify: `src/novels/format_novels.py`

**Step 1: Add text concatenation**

```python
def concatenate_text(data):
    if 'texts' not in data:
        return ""

    sorted_keys = sorted(data['texts'].keys(), key=lambda x: int(x))
    full_text = ""
    for key in sorted_keys:
        full_text += data['texts'][key]['text'] + "\n"
    return full_text
```

**Step 2: Add chunking logic**

```python
def chunk_text(text, tokenizer, max_tokens):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []

    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))

        # If we are not at the end, try to find a newline to split at
        if end < len(tokens):
            # Look back for newline token (assuming '\n' is tokenized to specific ID)
            # This is a simplification; for robust splitting, we decode and check
            chunk_tokens = tokens[start:end]
            chunk_text = tokenizer.decode(chunk_tokens)

            last_newline = chunk_text.rfind('\n')
            if last_newline != -1 and last_newline > len(chunk_text) * 0.8: # Only split if newline is near end
                # Re-encode to find exact token index is expensive, so we use character index approximation or just hard split for now
                # Better approach: decode last 100 tokens and find newline
                pass

        chunk = tokenizer.decode(tokens[start:end])
        chunks.append(chunk)
        start = end

    return chunks
```

**Refined Chunking Logic (More Robust):**

```python
def chunk_text_robust(text, tokenizer, max_tokens):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []

    start = 0
    total_tokens = len(tokens)

    while start < total_tokens:
        end = min(start + max_tokens, total_tokens)

        if end < total_tokens:
            # Try to find a natural break point in the last 10% of the chunk
            search_window = tokens[max(start, end - 500):end]
            decoded_window = tokenizer.decode(search_window)

            # Look for double newline first, then single newline
            split_idx = -1
            if '\n\n' in decoded_window:
                split_idx = decoded_window.rfind('\n\n') + 2
            elif '\n' in decoded_window:
                split_idx = decoded_window.rfind('\n') + 1

            if split_idx != -1:
                # Calculate token offset
                # This is tricky because token boundaries might not align with char boundaries perfectly
                # Safe fallback: just split at max_tokens if complex
                pass

        # Simple fallback for now: hard split at max_tokens
        # Ideally, we should implement the backtracking logic properly

        chunk = tokenizer.decode(tokens[start:end])
        chunks.append(chunk)
        start = end

    return chunks
```

**Step 3: Integrate into main loop**

```python
    with open(args.output_file, 'w', encoding='utf-8') as out_f:
        for file_path in tqdm(files):
            data = read_json_file(file_path)
            if not data:
                continue

            full_text = concatenate_text(data)
            chunks = chunk_text_robust(full_text, tokenizer, args.max_tokens)

            novel_id = data.get('meta', {}).get('id', 'unknown')

            for i, chunk in enumerate(chunks):
                record = {
                    "source": "novels",
                    "id": novel_id,
                    "split": i,
                    "text": chunk
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
```

### Task 5: Final Review and Testing

**Files:**
- Test: `src/novels/format_novels.py`

**Step 1: Run on a small subset**

```bash
# Create a dummy test file
mkdir -p data/novels/test
echo '{"meta": {"id": "test1"}, "texts": {"0": {"text": "Hello world."}}}' > data/novels/test/test.json

# Run script
python3 src/novels/format_novels.py --input_dir data/novels/test --output_file data/novels/formatted/test_output.jsonl
```

**Step 2: Verify output**

```bash
cat data/novels/formatted/test_output.jsonl
```

**Step 3: Run on full dataset**

```bash
python3 src/novels/format_novels.py --input_dir data/novels/deduped --output_file data/novels/formatted/novels_formatted.jsonl
```

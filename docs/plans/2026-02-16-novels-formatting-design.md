# Novels Data Formatting Design

## Overview
This document outlines the design for formatting the Light Novels dataset into a unified pre-training format suitable for the KawaiiLLM project.

## Goals
- Convert raw JSON files from `data/novels/deduped` into a single JSONL file.
- Ensure each text chunk does not exceed 4096 tokens.
- Use the Qwen tokenizer for accurate token counting.
- Respect paragraph boundaries (`\n\n`, `\n`) when splitting chunks.

## Input Format
The input files are JSON files located in `data/novels/deduped/<language>/<id>.json`.
Example structure:
```json
{
  "meta": {
    "id": "11860530",
    "title": "...",
    "language": "en",
    ...
  },
  "texts": {
    "0": {"text": "...", "words": ...},
    "1": {"text": "...", "words": ...},
    ...
  }
}
```

## Output Format
The output will be a JSONL file located at `data/novels/formatted/novels_formatted.jsonl`.
Each line will be a JSON object:
```json
{"source": "novels", "id": "11860530", "split": 0, "text": "chunk1_text"}
{"source": "novels", "id": "11860530", "split": 1, "text": "chunk2_text"}
...
```

## Implementation Details

### 1. Dependencies
- `transformers`: For `AutoTokenizer`.
- `torch`: Required by `transformers` for some models.
- `tqdm`: For progress bars.

### 2. Tokenizer
- Model: `Qwen/Qwen2.5-7B-Instruct` (or similar Qwen model).
- Loading: `AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)`.

### 3. Processing Logic
1. **Load Files**: Iterate through all JSON files in `data/novels/deduped`.
2. **Concatenate Text**: For each file, concatenate all `text` fields from the `texts` object in numerical order of keys.
3. **Tokenize**: Convert the full text into token IDs.
4. **Chunking**:
   - Iterate through token IDs.
   - Accumulate tokens up to 4096.
   - If a chunk exceeds 4096 tokens, backtrack to the last newline token (`\n` or `\n\n`) within a reasonable window (e.g., last 500 tokens).
   - If no newline found, force split at 4096.
   - Decode the chunk of token IDs back to text.
5. **Save**: Write the chunk to the output JSONL file.

### 4. Error Handling
- Skip files that cannot be parsed.
- Log warnings for files with empty content.
- Handle tokenizer errors gracefully.

## Future Considerations
- Parallel processing for faster execution on large datasets.
- Support for other datasets (MoeGirl, Bilibili) using similar logic.

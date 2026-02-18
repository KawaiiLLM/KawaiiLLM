# CLAUDE.md

This file provides guidance to Claude Code when working with the KawaiiLLM repository.

## Project Overview
KawaiiLLM is an ACGN-styled LLM with memory compression capabilities.
- **Goal**: ACGN role-playing and hierarchical memory compression.
- **Architecture**: MemE (Qwen3-Embedding-4B) -> Projector -> LLM (Qwen3-8B-Base).
- **Reference**: Based on [C3-Context-Cascade-Compression](https://github.com/liufanfanlff/C3-Context-Cascade-Compression).

## Environment Setup
```bash
conda create -n kawaii python=3.10 -y
conda activate kawaii
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers deepspeed accelerate
# Additional dependencies based on usage
pip install datasets pandas numpy
```

## Data Processing
Common commands for formatting data (run from project root):

### Novels
```bash
python src/novels/format_novels.py \
  --input_dir data/novels/deduped \
  --output_file data/novels/formatted/novels_formatted.jsonl
```

### Bilibili
```bash
python src/bilibili/format_bilibili.py \
  --input_path data/bilibili/cleaned \
  --output_file data/bilibili/formatted/bilibili_formatted.jsonl
```

### Merge and Shuffle
```bash
python src/merge_and_shuffle.py \
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

## Training Adaptation (C3 -> KawaiiLLM)
The training logic is adapted from `C3-Context-Cascade-Compression`.

### Key Differences
| Feature | C3 (Reference) | KawaiiLLM (Target) |
| :--- | :--- | :--- |
| **Encoder** | Separate LLM (e.g., Qwen2.5-1.5B) | Qwen3-Embedding-4B (MemE) |
| **Compression** | Q-Embeddings (Learnable Latent Tokens) | MemE Output + Projector |
| **Decoder** | Main LLM (e.g., Qwen2.5-3B) | Qwen3-8B-Base |
| **Training Goal** | Context Compression | Hierarchical Memory Compression |

### Adaptation Strategy
1.  **Model Definition**:
    - Replace `C3QwenModel`'s `llm1` with `Qwen3EmbeddingModel`.
    - Modify `mm_projector` to map MemE output dimension to LLM input dimension.
    - Implement "One Token" vs "Multiple Tokens" strategy in the forward pass.

2.  **Training Script**:
    - Adapt `C3-Context-Cascade-Compression/C3-master/C3/train/train.py`.
    - Ensure `DeepSpeed` configuration handles the multi-model setup (MemE + Projector + LLM).
    - Use `accelerate` for distributed training if needed.

### Reference Files
- **Model**: `C3-Context-Cascade-Compression/C3-master/C3/model/C3.py`
- **Training**: `C3-Context-Cascade-Compression/C3-master/C3/train/train.py`
- **Data Loading**: `C3-Context-Cascade-Compression/C3-master/C3/data/conversation_dataset_qwen.py`

## Code Style
- **Python**: Follow PEP 8.
- **Paths**: Use absolute paths for file operations.
- **Logging**: Use standard logging or `transformers.logging`.

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

## Training

### Architecture
- **MemE** (Qwen3-Embedding-4B): encoder, hidden_size=2560, 36 layers
- **Projector**: RMSNorm(2560) -> Linear(2560->10240) -> GELU -> Linear(10240->4096), plus learnable `mem_embeddings(128, 2560)`
- **LLM** (Qwen3-8B-Base): decoder, hidden_size=4096, 36 layers

### File Structure
```
src/train/
├── arguments.py       # ModelArguments, DataArguments, TrainingArguments
├── build_index.py     # Standalone: scan JSONL -> index, upsample, merge short orphans
├── model.py           # KawaiiLLMModel (MemE + Projector + LLM)
├── dataset.py         # KawaiiDataset with byte-offset access + deterministic task rotation
├── collator.py        # Left-pad context, right-pad target, collect per-sample n_mem
├── trainer.py         # KawaiiTrainer + callbacks (Curriculum, GradNorm, TaskLoss, NaN)
└── train.py           # Entry point
configs/
├── ds_zero2.json      # DeepSpeed ZeRO-2 config for 8x A800
└── train_8xa800.sh    # Launch script
scripts/
├── build_index.sh      # Script to build training index
└── train_multi_node.sh # Multi-node training launch script (torchrun + DeepSpeed)
```

### Training Pipeline

**Step 1: Build index** (run once after data formatting):
```bash
python src/train/build_index.py \
  --data_dirs data/novels/formatted data/bilibili/formatted \
    data/moegirl/formatted data/games/formatted \
    data/general/formatted data/math/formatted data/code/formatted \
  --output_path data/train_index.json \
  --upsample moegirl:3 \
  --merge_max_tokens 4000 --merge_short_threshold 2048
```

**Step 2: Launch training** (8x A800 80GB):
```bash
bash configs/train_8xa800.sh
```
Edit paths in `train_8xa800.sh` before running (`meme_model_name_or_path`, `llm_model_name_or_path`).

**Alternative: Multi-node Training** (using torchrun + DeepSpeed):
```bash
bash scripts/train_multi_node.sh
```
Environment variables `MASTER_ADDR`, `MASTER_PORT`, `NODE_NUM`, `GPU_NUM`, `RANK` are expected (or default to single-node).

### Key Design Decisions
- **Predecessor-based 2-task rotation**: Entries with a predecessor chunk: 2-task (reconstruction, continuation) via `(idx + epoch) % 2`. Entries without a predecessor: 2-task (NTP, reconstruction) via `(idx + epoch) % 2`. Continuation direction: prev_chunk → MemE context, current_chunk → LLM target.
- **Per-component LR**: projector+mem_embeddings ~5e-4, MemE ~1e-5, LLM ~1e-5.
- **MemE not frozen by default**: `freeze_meme=False` in launch script.
- **Flash Attention 2**: enabled for both MemE and LLM via `--attn_implementation flash_attention_2`.
- **Context left-padded** (MemE padding_side='left'), target right-padded (standard causal LM).
- **Checkpoints saved as**: `meme/`, `projector/` (projector.pt + mem_embeddings.pt), `llm/` subdirs.
- **Labels**: `cat([IGNORE * n_mem, target_labels])` — HF's internal shift handles alignment correctly.
- **Monitoring**: `--monitor_steps` controls per-component grad norm and per-task loss logging interval (default 10). GradNormCallback reports energy percentages (squared norm fractions) per optimizer step. TaskLossCallback reports mean loss for NTP/reconstruction/continuation tasks.

### C3 Reference (for context)
Adapted from `C3-Context-Cascade-Compression`. Key differences:

| Feature | C3 (Reference) | KawaiiLLM |
| :--- | :--- | :--- |
| **Encoder** | Separate LLM (Qwen2.5-1.5B) | Qwen3-Embedding-4B (MemE) |
| **Compression** | Q-Embeddings (learnable latent tokens) | MemE output + Projector |
| **Decoder** | Main LLM (Qwen2.5-3B) | Qwen3-8B-Base |
| **Encoder hidden** | varies | 2560 |
| **Decoder hidden** | varies | 4096 |

Reference files (read-only):
- `C3-Context-Cascade-Compression/C3-master/C3/model/C3.py`
- `C3-Context-Cascade-Compression/C3-master/C3/train/train.py`
- `C3-Context-Cascade-Compression/C3-master/C3/data/conversation_dataset_qwen.py`

## Code Style
- **Python**: Follow PEP 8.
- **Paths**: Use absolute paths for file operations.
- **Logging**: Use standard logging or `transformers.logging`.

#!/bin/bash
# Diagnostic run: use eager attention to test if Flash Attention causes NaN.
#
# Key changes vs train_debug_no_gc.sh:
#   --attn_implementation eager   (THE test: bypass Flash Attention entirely)
#   --gradient_checkpointing False (keep GC off as established)
#   --context_max_length 1024     (shorter — eager attention is O(n^2) memory)
#   --target_max_length 1024      (shorter)
#
# Eager attention computes full n×n attention matrix (no tiling/recomputation),
# which eliminates Flash Attention's bfloat16 backward precision issues.
#
# Memory estimate (per GPU, 8× A800 80GB, ZeRO-2):
#   MemE weights (bf16, frozen):      ~1 GB  (0.6B model)
#   LLM weights (bf16, full copy):    ~8 GB  (4B model)
#   LLM optimizer states (ZeRO-2):    ~4 GB
#   LLM gradients (ZeRO-2):           ~1 GB
#   LLM activations (no GC, seq≈1k): ~12 GB
#   Eager attn matrices (32 heads):   ~4 GB  (32 × 1k × 1k × 2 bytes)
#   Projector + misc:                  ~1 GB
#   Total:                           ~31 GB < 80 GB ✓
#
# Expected result:
#   - If NaN disappears → Flash Attention backward is the root cause
#   - If NaN persists   → root cause is elsewhere (RMSNorm, data, etc.)

set -euo pipefail

deepspeed --num_gpus 8 --module src.train.train \
  --deepspeed configs/ds_zero2.json \
  --meme_model_name_or_path /lpai/inputs/models/Qwen__Qwen3-Embedding-0.6B-main/ \
  --llm_model_name_or_path /lpai/inputs/models/Qwen__Qwen3-4B-Base-main/ \
  --attn_implementation eager \
  --data_dirs \
    data/novels/formatted \
    data/bilibili/formatted \
    data/moegirl/formatted \
    data/games/formatted \
    data/general/formatted \
    data/math/formatted \
    data/code/formatted \
  --index_path data/train_index.json \
  --output_dir /mnt/volumes/ss-sai-bd-ga/zhaoqixuan/output/kawaii_debug_eager_attn \
  --num_mem_tokens 128 \
  --freeze_meme True \
  --projector_lr 5e-4 \
  --llm_lr 1e-5 \
  --bf16 True \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 3 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --weight_decay 0.01 \
  --gradient_checkpointing False \
  --save_strategy no \
  --max_steps 50 \
  --logging_steps 1 \
  --dataloader_num_workers 32 \
  --report_to none \
  --run_name kawaii_debug_eager_attn

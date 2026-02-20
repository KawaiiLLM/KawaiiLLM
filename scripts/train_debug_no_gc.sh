#!/bin/bash
# Diagnostic run: disable gradient checkpointing to test if GC causes NaN.
#
# Key changes vs train_8xa800.sh:
#   --freeze_meme True              (no MemE gradients → saves ~16GB optimizer states)
#   --gradient_checkpointing False  (THE test: disable GC entirely)
#   --per_device_train_batch_size 1 (halved to fit activations without GC)
#   --gradient_accumulation_steps 16 (doubled to keep effective batch = 1*16*8 = 128)
#   --context_max_length 2048       (halved to reduce activation memory)
#   --target_max_length 2048        (halved to reduce activation memory)
#   --max_steps 50                  (short diagnostic run, just need step 0-2)
#
# Memory estimate (per GPU, 8x A800 80GB, ZeRO-2):
#   MemE weights (bf16, frozen, no optimizer):   ~8 GB
#   LLM weights (bf16, full copy):              ~16 GB
#   LLM optimizer states (ZeRO-2, /8):           ~8 GB
#   LLM gradients (ZeRO-2, /8):                  ~2 GB
#   LLM activations (no GC, batch=1, seq≈2k):  ~25 GB
#   Projector + misc:                             ~1 GB
#   Total:                                      ~60 GB < 80 GB ✓
#
# Expected result:
#   - If NaN disappears → GC is the root cause
#   - If NaN persists   → root cause is elsewhere (proceed to per-layer hooks)

set -euo pipefail

deepspeed --num_gpus 8 --module src.train.train \
  --deepspeed configs/ds_zero2.json \
  --meme_model_name_or_path /lpai/inputs/models/Qwen__Qwen3-Embedding-0.6B-main/ \
  --llm_model_name_or_path /lpai/inputs/models/Qwen__Qwen3-4B-Base-main/ \
  --data_dirs \
    data/novels/formatted \
    data/bilibili/formatted \
    data/moegirl/formatted \
    data/games/formatted \
    data/general/formatted \
    data/math/formatted \
    data/code/formatted \
  --index_path data/train_index.json \
  --output_dir /mnt/volumes/ss-sai-bd-ga/zhaoqixuan/output/kawaii_debug_no_gc \
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
  --run_name kawaii_debug_no_gc

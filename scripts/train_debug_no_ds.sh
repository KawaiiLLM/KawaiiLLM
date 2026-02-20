#!/bin/bash
# Diagnostic run: single GPU, no DeepSpeed.
#
# Hypothesis: DeepSpeed ZeRO-2 bf16 gradient allreduce overflows to inf on 8 GPUs
# (NCCL sums 8 × large_grad in bf16 → exceeds bf16 max 65504 → inf), which
# DeepSpeed's gradient clipping then converts to NaN (inf × 0 = NaN, IEEE 754).
# Single GPU eliminates all allreduce operations entirely.
#
# Key changes vs train_debug_no_gc.sh:
#   - No deepspeed launcher, no --deepspeed flag
#   - torchrun --nproc_per_node=1  (single GPU, no allreduce)
#   - gradient_accumulation_steps 1  (no accumulation needed)
#   - context_max_length 512  (keep activations small without ZeRO)
#   - target_max_length 512   (keep activations small without ZeRO)
#   - max_steps 5             (only need to observe step 0-1)
#
# Memory estimate (1x A800 80GB, no ZeRO):
#   MemE weights (bf16, frozen, no optimizer):   ~1 GB
#   LLM weights (bf16):                          ~8 GB
#   LLM Adam moments (fp32, m+v):               ~28 GB
#   LLM gradients (bf16):                        ~7 GB
#   Projector + mem_embeddings + misc:            ~1 GB
#   Activations (no GC, batch=1, seq≈512):       ~5 GB
#   Total:                                      ~50 GB < 80 GB ✓
#
# Expected result:
#   - If NaN disappears → allreduce is root cause → fix: communication_data_type=fp32
#   - If NaN persists   → root cause is not DeepSpeed-specific (deeper issue)

set -euo pipefail

torchrun --nproc_per_node=1 -m src.train.train \
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
  --output_dir /mnt/volumes/ss-sai-bd-ga/zhaoqixuan/output/kawaii_debug_no_ds \
  --num_mem_tokens 128 \
  --freeze_meme True \
  --projector_lr 5e-4 \
  --llm_lr 1e-5 \
  --bf16 True \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 3 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --weight_decay 0.01 \
  --gradient_checkpointing False \
  --save_strategy no \
  --max_steps 5 \
  --logging_steps 1 \
  --dataloader_num_workers 4 \
  --report_to none \
  --run_name kawaii_debug_no_ds \
  --context_max_length 512 \
  --target_max_length 512

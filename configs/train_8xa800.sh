#!/bin/bash
# KawaiiLLM training launch script for 8x A800 80GB
# Effective batch size: 2 * 8 * 8 = 128

set -euo pipefail

deepspeed --num_gpus 8 src/train/train.py \
  --deepspeed configs/ds_zero2.json \
  --meme_model_name_or_path /path/to/Qwen3-Embedding-4B \
  --llm_model_name_or_path /path/to/Qwen3-8B-Base \
  --data_dirs \
    data/novels/formatted \
    data/bilibili/formatted \
    data/moegirl/formatted \
    data/games/formatted \
    data/general/formatted \
    data/math/formatted \
    data/code/formatted \
  --index_path data/train_index.json \
  --output_dir output/kawaii_v1 \
  --num_mem_tokens 128 \
  --freeze_meme False \
  --projector_lr 1e-3 \
  --meme_lr 1e-6 \
  --llm_lr 1e-6 \
  --bf16 True \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 3 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --weight_decay 0.01 \
  --gradient_checkpointing True \
  --save_strategy steps \
  --save_steps 2000 \
  --save_total_limit 3 \
  --logging_steps 10 \
  --dataloader_num_workers 4 \
  --report_to wandb \
  --run_name kawaii_v1

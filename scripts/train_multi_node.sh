#!/bin/bash
# KawaiiLLM multi-node training launch script
#
# Uses torchrun + DeepSpeed (as library) for multi-node distributed training.
# Environment variables are injected by the cluster scheduler:
#   MASTER_ADDR  — master node IP
#   MASTER_PORT  — communication port
#   NODE_NUM     — total number of nodes
#   GPU_NUM      — GPUs per node
#   RANK         — current node rank (0-based)
#
# Single-node fallback: all default to localhost / 1 node / 8 GPUs / rank 0.
#
# Effective batch size = per_device_batch * GPU_NUM * NODE_NUM * grad_accum
# Default: 2 * 8 * 2 * 4 = 128 (2 nodes, grad_accum adjusted to keep total=128)
# Adjust --gradient_accumulation_steps when changing node count.
#
# Note: --learning_rate is the scheduler base reference only;
# all param groups use explicit LRs (projector_lr, meme_lr, llm_lr).

set -euo pipefail

#==============================================================================
# Distributed environment (injected by cluster scheduler, with sane defaults)
#==============================================================================
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
NODE_NUM="${NODE_NUM:-1}"
GPU_NUM="${GPU_NUM:-8}"
RANK="${RANK:-0}"

echo "=== Multi-node config ==="
echo "  MASTER_ADDR : ${MASTER_ADDR}"
echo "  MASTER_PORT : ${MASTER_PORT}"
echo "  NODE_NUM    : ${NODE_NUM}"
echo "  GPU_NUM     : ${GPU_NUM}"
echo "  RANK        : ${RANK}"
echo "========================="

#==============================================================================
# Launch: torchrun + DeepSpeed as library (--deepspeed flag handled by HF Trainer)
#==============================================================================
torchrun \
  --master-addr="${MASTER_ADDR}" \
  --master-port="${MASTER_PORT}" \
  --nnodes="${NODE_NUM}" \
  --nproc-per-node="${GPU_NUM}" \
  --node_rank="${RANK}" \
  -m src.train.train \
  --deepspeed configs/ds_zero2.json \
  --meme_model_name_or_path /lpai/inputs/models/Qwen__Qwen3-Embedding-4B-main/ \
  --llm_model_name_or_path /lpai/inputs/models/qwen__qwen3-8b-base-25-04-28-1833/ \
  --data_dirs \
    data/novels/formatted \
    data/bilibili/formatted \
    data/moegirl/formatted \
    data/games/formatted \
    data/general/formatted \
    data/math/formatted \
    data/code/formatted \
  --index_path data/train_index.json \
  --val_index_path data/train_index_val.json \
  --output_dir /mnt/volumes/ss-sai-bd-ga/zhaoqixuan/output/kawaii_4b_8b \
  --num_mem_tokens 128 \
  --attn_implementation flash_attention_2 \
  --freeze_meme False \
  --projector_lr 1e-4 \
  --meme_lr 2e-5 \
  --llm_lr 1e-5 \
  --bf16 True \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 2 \
  --learning_rate 1e-5 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --weight_decay 0.01 \
  --tf32 True \
  --gradient_checkpointing True \
  --save_strategy steps \
  --save_steps 5000 \
  --save_total_limit 5 \
  --per_device_eval_batch_size 2 \
  --eval_strategy steps \
  --eval_steps 500 \
  --prediction_loss_only True \
  --logging_steps 1 \
  --monitor_steps 10 \
  --dataloader_num_workers 32 \
  --logging_dir /lpai/output/tensorboard \
  --report_to tensorboard \
  --run_name kawaii_4b_8b

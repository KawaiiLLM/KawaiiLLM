#!/bin/bash
# Build byte-offset index over all formatted JSONL files.
# Run from project root: bash scripts/build_index.sh

set -euo pipefail

python -m src.train.build_index \
  --data_dirs \
    data/novels/formatted \
    data/bilibili/formatted \
    data/moegirl/formatted \
    data/games/formatted \
    data/general/formatted \
    data/math/formatted \
    data/code/formatted \
  --output_path data/train_index.json \
  --val_ratio 0.001 \
  --val_output_path data/train_index_val.json \
  --test_ratio 0.009 \
  --test_output_path data/train_index_test.json \
  --upsample moegirl:3 \
  --merge_max_tokens 4000 \
  --merge_short_threshold 2048

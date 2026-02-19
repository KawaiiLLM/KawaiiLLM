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
  --upsample moegirl:3 \
  --merge_max_tokens 3500 \
  --merge_short_threshold 2048

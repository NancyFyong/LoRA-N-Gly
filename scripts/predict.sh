#!/usr/bin/env bash
set -euo pipefail

python main.py \
  --stage test \
  --model_name facebook/esm2_t36_3B_UR50D \
  --train_dataset ./data/N-GlycositeAltas/train.csv \
  --valid_dataset ./data/N-GlycositeAltas/valid.csv \
  --test_dataset ./data/N-GlycositeAltas/test.csv \
  --peft_model_path ./checkpoints/N-linked/ESM-3B/checkpoint

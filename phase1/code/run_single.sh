#!/bin/bash
# Memory-frugal single-model run for 8GB M1: head and fine-tune as SEPARATE
# processes so all memory is released between phases (defeats the TF-Metal leak).
cd /Users/suharshkumar/Documents/claude/ddp_summer_dummy/run
source .venv/bin/activate

export BACKBONE=B0 IMG_SIZE=96 BATCH_SIZE=16 GAMMA=0 CW_POWER=0 LABEL_SMOOTH=0.05
export DROPOUT=0.40 UNFREEZE_TOP=40 LR_HEAD=1e-3 LR_FT=1e-4 TTA_N=12
export OUT_DIR=artifacts/single_b0

echo "######## PHASE HEAD (separate process) ########"
MODE=head EPOCHS_HEAD=15 PAT_HEAD=5 python -u train.py
echo "HEAD_EXIT=$?"

echo "######## PHASE FINETUNE (separate process) ########"
MODE=finetune EPOCHS_FT=30 PAT_FT=6 python -u train.py
echo "FINETUNE_EXIT=$?"

echo "ALL_DONE"

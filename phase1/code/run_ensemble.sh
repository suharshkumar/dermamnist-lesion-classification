#!/bin/bash
# Sequentially train top-1-optimized models for ensembling.
cd /Users/suharshkumar/Documents/claude/ddp_summer_dummy/run
source .venv/bin/activate

COMMON="IMG_SIZE=128 BATCH_SIZE=32 GAMMA=0 CW_POWER=0 LABEL_SMOOTH=0.05 DROPOUT=0.40 \
EPOCHS_HEAD=18 PAT_HEAD=6 LR_HEAD=1e-3 \
EPOCHS_FT=45 PAT_FT=10 LR_FT=1e-4 UNFREEZE_TOP=100 TTA_N=15"

echo "######## MODEL 1: EfficientNetB0 ########"
env $COMMON BACKBONE=B0   OUT_DIR=artifacts/m1_b0   python train.py
echo "MODEL1_EXIT=$?"

echo "######## MODEL 2: EfficientNetV2B0 ########"
env $COMMON BACKBONE=V2B0 OUT_DIR=artifacts/m2_v2b0 python train.py
echo "MODEL2_EXIT=$?"

echo "ALL_MODELS_DONE"

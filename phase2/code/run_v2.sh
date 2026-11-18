#!/bin/bash
# Phase-2 pipeline (Nov 2026). Each stage is its own process so TF-Metal memory is
# released between stages (8 GB M1). Logs: logs_v2/*.log. Safe to re-run: finished
# stages are skipped when their results.json exists.
cd /Users/suharshkumar/Documents/claude/ddp_summer_dummy/run
source .venv/bin/activate
mkdir -p logs_v2
export IMG_SIZE=96 BATCH_SIZE=16 GAMMA=0 CW_POWER=0 LABEL_SMOOTH=0.05 DROPOUT=0.40 TTA_N=8
stage () {   # name, then env assignments
  local name=$1; shift
  if [ -f "artifacts/$name/results.json" ] && [ "$name" != "single_b0" -o -f artifacts/single_b0/val_tta_probs.npy ]; then
    echo "== skip $name (done)"; return; fi
  echo "== $(date '+%F %T') start $name"
  env "$@" OUT_DIR=artifacts/$name python -u train_v2.py > logs_v2/$name.log 2>&1
  echo "== $(date '+%F %T') end $name exit=$?"
}
# 0. baseline checkpoint: regenerate val+test probabilities (needed for val-tuned ensemble)
stage single_b0      MODE=eval BACKBONE=B0
# 1. resume fine-tuning of the 80.0% checkpoint (its run was cut at epoch 10/30 with val acc still rising)
stage v2_b0_resume   MODE=resume BACKBONE=B0 INIT_MODEL=artifacts/single_b0/best_model.keras \
                     UNFREEZE_TOP=60 LR_FT=5e-5 LR_FT_MIN=1e-6 EPOCHS_FT=20 PAT_FT=8
# 2. second ensemble member: EfficientNetV2-B0 (head, then fine-tune, separate processes)
if [ ! -f artifacts/v2_v2b0/results.json ]; then
  echo "== $(date '+%F %T') start v2_v2b0 head"
  env MODE=head BACKBONE=V2B0 EPOCHS_HEAD=12 PAT_HEAD=4 OUT_DIR=artifacts/v2_v2b0 python -u train_v2.py > logs_v2/v2_v2b0_head.log 2>&1
  echo "== $(date '+%F %T') head exit=$?"
fi
stage v2_v2b0        MODE=finetune BACKBONE=V2B0 UNFREEZE_TOP=60 LR_FT=1e-4 EPOCHS_FT=20 PAT_FT=7
# 3. ensemble + prior correction
echo "== $(date '+%F %T') ensemble"
python -u ensemble_v2.py artifacts/single_b0 artifacts/v2_b0_resume artifacts/v2_v2b0 > logs_v2/ensemble.log 2>&1
echo "== $(date '+%F %T') ALL_DONE exit=$?"

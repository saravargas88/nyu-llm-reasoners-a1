#!/bin/bash
# h200_lr_refine_then_bs64_128.sh
# Goal: refine LR around 1e-4 (your current best), then test batch sizes 64 vs 128.
# H200 / CUDA assumed.

BASE="uv run python student/train.py"

TRAIN_PATH="/home/ql2221/train_tokens.npy"
VAL_PATH="/home/ql2221/valid_tokens.npy"
CKPT_DIR="/scratch/ql2221/checkpoints_h200"
mkdir -p "$CKPT_DIR"

DEVICE="cuda"

MODEL="--vocab_size 10000 \
       --context_length 256 \
       --d_model 512 \
       --d_ff 1344 \
       --num_layers 4 \
       --num_heads 16 \
       --theta 10000"

COMMON="--train_path $TRAIN_PATH \
        --val_path $VAL_PATH \
        --checkpoint_dir $CKPT_DIR \
        --device $DEVICE \
        --weight_decay 0.1 \
        --beta1 0.9 \
        --beta2 0.999 \
        --epsilon 1e-8 \
        --grad_clip_max_l2_norm 1.0 \
        --eval_steps 20"

# ------------------------------------------------------------------------------
# PHASE 1: LR refinement probe @ bs=64 (fast)
# ------------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " PHASE 1: LR refinement probe @ bs=64"
echo "============================================================"

BS=64
PROBE_STEPS=3000
PROBE_WARMUP=300

# around best (1e-4)
LRS_PROBE=(7e-5 1e-4 1.5e-4 2e-4)

for LR in "${LRS_PROBE[@]}"; do
  LR_MIN=$(python3 - <<PY
lr=float("$LR")
print(f"{lr/10:.0e}")
PY
)
  echo ">>> [probe] bs=$BS steps=$PROBE_STEPS lr=$LR lr_min=$LR_MIN  [$(date)]"
  $BASE \
    $MODEL $COMMON \
    --batch_size $BS \
    --total_steps $PROBE_STEPS \
    --warmup_steps $PROBE_WARMUP \
    --lr "$LR" \
    --lr_min "$LR_MIN" \
    --wandb_project "a1_h200_lr_refine_probe" \
    --run_name "lr_refine_probe__bs=${BS}__steps=${PROBE_STEPS}__lr=${LR}"
done

echo ""
echo ">>> Pick BEST_LR from wandb project: a1_h200_lr_refine_probe"
echo ">>> Then set BEST_LR / BEST_LR_MIN below and run the rest."
echo ""

# ------------------------------------------------------------------------------
# SET THESE AFTER YOU INSPECT PROBES
# ------------------------------------------------------------------------------
BEST_LR="1e-4"     # <-- EDIT after probes
BEST_LR_MIN="1e-5" # <-- should be BEST_LR/10

# ------------------------------------------------------------------------------
# PHASE 2: Full confirmation run @ bs=64 (deliverable baseline)
# ------------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " PHASE 2: Full confirmation @ bs=64"
echo "============================================================"

FULL_STEPS=5000
FULL_WARMUP=500

echo ">>> [full] bs=64 steps=$FULL_STEPS lr=$BEST_LR lr_min=$BEST_LR_MIN  [$(date)]"
$BASE \
  $MODEL $COMMON \
  --batch_size 64 \
  --total_steps $FULL_STEPS \
  --warmup_steps $FULL_WARMUP \
  --lr "$BEST_LR" \
  --lr_min "$BEST_LR_MIN" \
  --wandb_project "a1_h200_lr_confirm" \
  --run_name "lr_confirm__bs=64__steps=${FULL_STEPS}__lr=${BEST_LR}"

# ------------------------------------------------------------------------------
# PHASE 3: Batch size test (64 vs 128) with LR re-tuning for 128
# ------------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " PHASE 3: Batch size test (64 vs 128) with LR retune"
echo "============================================================"

# bs=64 re-run (optional if you want directly comparable curves in same project)
echo ">>> [bs test] bs=64 @ BEST_LR  [$(date)]"
$BASE \
  $MODEL $COMMON \
  --batch_size 64 \
  --total_steps $FULL_STEPS \
  --warmup_steps $FULL_WARMUP \
  --lr "$BEST_LR" \
  --lr_min "$BEST_LR_MIN" \
  --wandb_project "a1_h200_batch_64_128" \
  --run_name "batch_test__bs=64__lr=${BEST_LR}"

# bs=128: try same LR and 2x LR (common heuristic), keep lr_min = lr/10
for MULT in 1.0 2.0; do
  LR_128=$(python3 - <<PY
base=float("$BEST_LR")
m=float("$MULT")
print(f"{base*m:.2e}")
PY
)
  LR_128_MIN=$(python3 - <<PY
lr=float("$LR_128")
print(f"{lr/10:.0e}")
PY
)

  # warmup a touch longer for bigger batch (optional but often helps)
  WARMUP_128=600

  echo ">>> [bs test] bs=128 lr=$LR_128 lr_min=$LR_128_MIN warmup=$WARMUP_128  [$(date)]"
  $BASE \
    $MODEL $COMMON \
    --batch_size 128 \
    --total_steps $FULL_STEPS \
    --warmup_steps $WARMUP_128 \
    --lr "$LR_128" \
    --lr_min "$LR_128_MIN" \
    --wandb_project "a1_h200_batch_64_128" \
    --run_name "batch_test__bs=128__lr=${LR_128}__warmup=${WARMUP_128}"
done

echo ""
echo "============================================================"
echo " DONE"
echo " Projects:"
echo "  - a1_h200_lr_refine_probe : Phase 1"
echo "  - a1_h200_lr_confirm      : Phase 2"
echo "  - a1_h200_batch_64_128    : Phase 3"
echo "============================================================"

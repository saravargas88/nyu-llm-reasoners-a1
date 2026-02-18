#!/bin/bash
# h200_lr_refine_then_bs64_128.sh
# Goal: refine LR around 1e-4 (your current best), then test batch sizes 64 vs 128.
# H200 / CUDA assumed.

BASE="uv run python -m student.train"

TRAIN_PATH="/home/ql2221/Projects/train_tokens.npy"
VAL_PATH="/home/ql2221/Projects/valid_tokens.npy"
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
        --eval_steps 20
        --resume_from_checkpoint <dir>"


BEST_LR="2e-4"

# ------------------------------------------------------------------------------
# BEST CHECKPOINT PATH 
# ------------------------------------------------------------------------------
BEST_CKPT="$CKPT_DIR/lr_refine_probe__bs=64__steps=3000__lr=${BEST_LR}/last.pt"


# ------------------------------------------------------------------------------
# PHASE 2: Continue from BEST 3k checkpoint
# ------------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " PHASE 2: Resume best run and extend training"
echo "============================================================"

EXTENDED_STEPS=20000      # total steps you want to reach
RESUME_WARMUP=0          # no need to warmup again if continuing schedule

echo ">>> [resume] bs=64 extend_to=$EXTENDED_STEPS from lr=$BEST_LR  [$(date)]"

$BASE \
  $MODEL $COMMON \
  --batch_size 64 \
  --total_steps $EXTENDED_STEPS \
  --warmup_steps $RESUME_WARMUP \
  --lr "$BEST_LR" \
  --lr_min "$(python3 - <<PY
lr=float("$BEST_LR")
print(f"{lr/10:.0e}")
PY
)" \
  --resume_from_checkpoint "$BEST_CKPT" \
  --wandb_project "a1_h200_extended" \
  --run_name "extended_from_3k__lr=${BEST_LR}__to_${EXTENDED_STEPS}"

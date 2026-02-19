#!/usr/bin/env bash
set -euo pipefail

# ---- user-tuned defaults ----
PROJECT="ablations-llm-reasoners-a1"
CKPT_DIR="/scratch/ql2221/checkpoints_h200"
TRAIN_PATH="/home/ql2221/Projects/train_tokens.npy"
VAL_PATH="/home/ql2221/Projects/valid_tokens.npy"

# optimal LR you found
LR=1e-5

# keep cosine schedule shape similar; make lr_min smaller but same order
# (if you want strictly constant LR, set LR_MIN=1e-5 and warmup_steps=0)
LR_MIN=1e-6

# training budget knobs (match your assignment budget)
TOTAL_STEPS=5000
WARMUP_STEPS=200
BATCH_SIZE=64

# model config (must match your baseline)
VOCAB_SIZE=10000
CTX=256
D_MODEL=512
D_FF=1344
N_LAYERS=4
N_HEADS=16
THETA=10000

# gradient clip
CLIP=1.0

# weight decay/betas (match your defaults unless you intentionally change)
WD=0.1
B1=0.9
B2=0.999
EPS=1e-8

STAMP="$(date +%Y%m%d_%H%M%S)"

run () {
  local name="$1"; shift
  echo "=== Running: ${name} ==="
  python train.py \
    --wandb_project "${PROJECT}" \
    --run_name "${name}" \
    --checkpoint_dir "${CKPT_DIR}" \
    --train_path "${TRAIN_PATH}" \
    --val_path "${VAL_PATH}" \
    --lr "${LR}" \
    --lr_min "${LR_MIN}" \
    --total_steps "${TOTAL_STEPS}" \
    --warmup_steps "${WARMUP_STEPS}" \
    --batch_size "${BATCH_SIZE}" \
    --vocab_size "${VOCAB_SIZE}" \
    --context_length "${CTX}" \
    --d_model "${D_MODEL}" \
    --d_ff "${D_FF}" \
    --num_layers "${N_LAYERS}" \
    --num_heads "${N_HEADS}" \
    --theta "${THETA}" \
    --grad_clip_max_l2_norm "${CLIP}" \
    --beta1 "${B1}" \
    --beta2 "${B2}" \
    --epsilon "${EPS}" \
    --weight_decay "${WD}" \
    "$@"
}

# -----------------------------
# 0) Baseline (pre-norm + RoPE + SwiGLU + RMSNorm)
# -----------------------------
#run "baseline_prenorm_rope_swiglu_rmsnorm__lr${LR}__${STAMP}"

# -----------------------------
# 1) Post-norm ablation
# -----------------------------
run "ablation_postnorm__lr${LR}__${STAMP}" \
  --post_norm

# -----------------------------
# 2) No position embeddings (NoPE) ablation
# -----------------------------
run "ablation_nope_no_rope__lr${LR}__${STAMP}" \
  --no_rope

# -----------------------------
# 3) SwiGLU vs SiLU ablation (SiLU = no gating)
# NOTE: your assignment says d_ff should change to 4*d_model to match params.
# Your train.py currently passes --d_ff directly; so we override d_ff here.
# For D_MODEL=512 -> 4*d_model = 2048.
# -----------------------------
run "ablation_silu_ffn__lr${LR}__${STAMP}" \
  --use_silu \
  --d_ff 2048

# -----------------------------
# 4) Remove RMSNorm ablation
#   a) try at optimal lr=1e-5 (likely unstable)
#   b) stability sweep at lower lrs (assignment asks)
# We keep the same schedule ratio by scaling lr_min with lr.
# -----------------------------
run "ablation_no_rmsnorm__lr${LR}__${STAMP}" \
  --no_rmsnorm

for LOW_LR in 5e-6 2e-6 1e-6; do
  # pick lr_min = lr/10 for consistency
  LOW_MIN="$(python - <<'PY'
import sys
lr=float(sys.argv[1])
print(f"{lr/10:.1e}")
PY
"${LOW_LR}")"

  python student/train.py \
    --wandb_project "${PROJECT}" \
    --run_name "ablation_no_rmsnorm__lr${LOW_LR}__${STAMP}" \
    --checkpoint_dir "${CKPT_DIR}" \
    --train_path "${TRAIN_PATH}" \
    --val_path "${VAL_PATH}" \
    --lr "${LOW_LR}" \
    --lr_min "${LOW_MIN}" \
    --total_steps "${TOTAL_STEPS}" \
    --warmup_steps "${WARMUP_STEPS}" \
    --batch_size "${BATCH_SIZE}" \
    --vocab_size "${VOCAB_SIZE}" \
    --context_length "${CTX}" \
    --d_model "${D_MODEL}" \
    --d_ff "${D_FF}" \
    --num_layers "${N_LAYERS}" \
    --num_heads "${N_HEADS}" \
    --theta "${THETA}" \
    --grad_clip_max_l2_norm "${CLIP}" \
    --beta1 "${B1}" \
    --beta2 "${B2}" \
    --epsilon "${EPS}" \
    --weight_decay "${WD}" \
    --no_rmsnorm
done

echo "All ablations launched."

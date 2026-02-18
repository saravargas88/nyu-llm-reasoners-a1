#!/bin/bash
# run_overnight_mps.sh
# Full overnight run on Apple Silicon MPS — all required deliverables.
# Token budget: 32 x 5000 x 256 = 40,960,000 (per assignment MPS spec).
# Target val loss: ≤ 2.00 (MPS downscaled target per spec).
# Estimated total time: ~5-7 hours on M3 Max.
#
# Deliverables covered:
#   [1] LR sweep           → find best LR, hit val loss ≤ 2.00
#   [2] Baseline (best LR) → checkpoint for generation + ablation reference
#   [3] no_rmsnorm         → layer norm ablation (best LR + low LR)
#   [4] post_norm          → pre vs post norm ablation
#   [5] no_rope            → NoPE ablation
#   [6] silu               → SwiGLU vs SiLU ablation
#   [7] batch sweep        → bs ∈ {1, 8, 32, 64, 128} scaled to 40M tokens
#
# MPS-specific notes per assignment spec:
#   - Do NOT set torch.set_float32_matmul_precision('high') — silently broken on MPS
#   - torch.compile with backend="aot_eager" is safe on MPS (Inductor is NOT)
#   - Total tokens fixed at 40,960,000 across all runs
#   - Cosine decay terminates exactly at total_steps (no tail at lr_min)
#
# Usage:
#   tmux new -s overnight
#   bash run_overnight_mps.sh 2>&1 | tee overnight_mps.log

set -e

BASE="uv run python -m student.train"

# ── Paths ─────────────────────────────────────────────────────────────────────
TRAIN_PATH="/Users/sara/Desktop/SPRING2026/LLM Reasoners/nyu-llm-reasoners-a1/data/results/train_tokens.npy"
VAL_PATH="/Users/sara/Desktop/SPRING2026/LLM Reasoners/nyu-llm-reasoners-a1/data/results/valid_tokens.npy"
CKPT_DIR="/Users/sara/Desktop/SPRING2026/LLM Reasoners/nyu-llm-reasoners-a1/data/checkpoints"

mkdir -p "$CKPT_DIR"

# ── Fixed model config per spec (~22.7M params) ───────────────────────────────
MODEL="--vocab_size 10000 \
       --context_length 256 \
       --d_model 512 \
       --d_ff 1344 \
       --num_layers 4 \
       --num_heads 16 \
       --theta 10000"

# ── MPS standard training config — 40M token budget ──────────────────────────
# 32 * 5000 * 256 = 40,960,000 tokens (per assignment MPS spec).
# warmup = 10% of 5000 = 500 steps.
# Cosine decay runs from step 0 to exactly step 5000 (Tc = total_steps).
MPS_TRAIN="--batch_size 32 \
           --total_steps 5000 \
           --warmup_steps 500 \
           --weight_decay 0.1 \
           --beta1 0.9 \
           --beta2 0.999 \
           --epsilon 1e-8 \
           --grad_clip_max_l2_norm 1.0 \
           --eval_steps 20 \
           --device mps "

echo "============================================================"
echo " OVERNIGHT RUN — Apple Silicon MPS"
echo " Started: $(date)"
echo " Token budget: 32 x 5000 x 256 = 40,960,000"
echo " Target val loss: ≤ 2.00"
echo "============================================================"


# ════════════════════════════════════════════════════════════════
# PHASE 1 — LR SWEEP (~1.5 hrs, 3 runs)
#
# Same candidates as H200 version — the optimal LR is largely
# model/architecture dependent, not hardware dependent.
# lr_min = lr/10 throughout (correct cosine decay ratio).
# ════════════════════════════════════════════════════════════════
echo ""
echo ">>> PHASE 1: LR Sweep (3 runs)  [$(date)]"

for LR in 3e-4 5e-4 8e-4; do
    LR_MIN=$(python3 -c "print($LR / 10)")
    echo "  → lr=$LR  lr_min=$LR_MIN  [$(date)]"
    $BASE \
        --train_path "$TRAIN_PATH" \
        --val_path   "$VAL_PATH" \
        $MODEL $MPS_TRAIN \
        --lr     $LR \
        --lr_min $LR_MIN \
        --wandb_project "llm-reasoners-a1-lr-sweep-mps" \
        --run_name "lr_sweep__lr=${LR}"
done

echo ">>> PHASE 1 complete: $(date)"
echo "    Check wandb: llm-reasoners-a1-lr-sweep-mps"


# ════════════════════════════════════════════════════════════════
# PHASE 2 — BASELINE at best LR (~36 min per spec)
#
# 5e-4 hardcoded as expected best. This is your generation
# checkpoint and ablation reference curve.
# ════════════════════════════════════════════════════════════════
echo ""
echo ">>> PHASE 2: Baseline at best LR=5e-4  [$(date)]"

BEST_LR=5e-4
BEST_LR_MIN=5e-5

$BASE \
    --train_path "$TRAIN_PATH" \
    --val_path   "$VAL_PATH" \
    $MODEL $MPS_TRAIN \
    --lr     $BEST_LR \
    --lr_min $BEST_LR_MIN \
    --wandb_project "llm-reasoners-a1-ablations-mps" \
    --run_name "ablation__baseline"

echo ">>> PHASE 2 complete: $(date)"


# ════════════════════════════════════════════════════════════════
# PHASE 3 — ABLATIONS (~3 hrs, 5 runs)
# ════════════════════════════════════════════════════════════════
echo ""
echo ">>> PHASE 3: Ablations (5 runs)  [$(date)]"

# ── 3a. No RMSNorm at best LR (expect instability) ────────────
echo "  → [1/5] no_rmsnorm @ best LR  [$(date)]"
$BASE \
    --train_path "$TRAIN_PATH" \
    --val_path   "$VAL_PATH" \
    $MODEL $MPS_TRAIN \
    --lr     $BEST_LR \
    --lr_min $BEST_LR_MIN \
    --no_rmsnorm \
    --wandb_project "llm-reasoners-a1-ablations-mps" \
    --run_name "ablation__no_rmsnorm"

# ── 3b. No RMSNorm at lower LR (attempt to stabilise) ─────────
echo "  → [2/5] no_rmsnorm @ low LR=1e-4  [$(date)]"
$BASE \
    --train_path "$TRAIN_PATH" \
    --val_path   "$VAL_PATH" \
    $MODEL $MPS_TRAIN \
    --lr     1e-4 \
    --lr_min 1e-5 \
    --no_rmsnorm \
    --wandb_project "llm-reasoners-a1-ablations-mps" \
    --run_name "ablation__no_rmsnorm_lowlr"

# ── 3c. Post-norm ──────────────────────────────────────────────
echo "  → [3/5] post_norm  [$(date)]"
$BASE \
    --train_path "$TRAIN_PATH" \
    --val_path   "$VAL_PATH" \
    $MODEL $MPS_TRAIN \
    --lr     $BEST_LR \
    --lr_min $BEST_LR_MIN \
    --post_norm \
    --wandb_project "llm-reasoners-a1-ablations-mps" \
    --run_name "ablation__post_norm"

# ── 3d. NoPE ──────────────────────────────────────────────────
echo "  → [4/5] no_rope (NoPE)  [$(date)]"
$BASE \
    --train_path "$TRAIN_PATH" \
    --val_path   "$VAL_PATH" \
    $MODEL $MPS_TRAIN \
    --lr     $BEST_LR \
    --lr_min $BEST_LR_MIN \
    --no_rope \
    --wandb_project "llm-reasoners-a1-ablations-mps" \
    --run_name "ablation__no_rope"

# ── 3e. SiLU vs SwiGLU ────────────────────────────────────────
echo "  → [5/5] silu vs swiglu  [$(date)]"
$BASE \
    --train_path "$TRAIN_PATH" \
    --val_path   "$VAL_PATH" \
    $MODEL $MPS_TRAIN \
    --lr     $BEST_LR \
    --lr_min $BEST_LR_MIN \
    --use_silu \
    --wandb_project "llm-reasoners-a1-ablations-mps" \
    --run_name "ablation__silu"

echo ">>> PHASE 3 complete: $(date)"


# ════════════════════════════════════════════════════════════════
# PHASE 4 — BATCH SIZE SWEEP (~1.5 hrs, 5 runs)
#
# Token budget stays fixed at 40,960,000.
# total_steps = 40,960,000 / (bs * 256), warmup = 10% of steps.
# bs=256 skipped — at 40M tokens that's only ~625 steps, too few
# to be meaningful. bs=1 runs last as it's the slowest by far.
# ════════════════════════════════════════════════════════════════
echo ""
echo ">>> PHASE 4: Batch Size Sweep (5 runs)  [$(date)]"

for BS in 8 32 64 128 1; do
    STEPS=$(python3 -c "print(int(40_960_000 / ($BS * 256)))")
    WARMUP=$(python3 -c "print(max(100, int($STEPS * 0.10)))")
    echo "  → bs=$BS  steps=$STEPS  warmup=$WARMUP  [$(date)]"
    $BASE \
        --train_path "$TRAIN_PATH" \
        --val_path   "$VAL_PATH" \
        $MODEL \
        --batch_size   $BS \
        --total_steps  $STEPS \
        --warmup_steps $WARMUP \
        --lr           $BEST_LR \
        --lr_min       $BEST_LR_MIN \
        --weight_decay 0.1 \
        --beta1 0.9 \
        --beta2 0.999 \
        --epsilon 1e-8 \
        --grad_clip_max_l2_norm 1.0 \
        --eval_steps 20 \
        --device mps \
        --checkpoint_dir $CKPT_DIR \
        --wandb_project "llm-reasoners-a1-batch-sweep-mps" \
        --run_name "batch_sweep__bs=${BS}"
done

echo ">>> PHASE 4 complete: $(date)"


echo ""
echo "============================================================"
echo " ALL PHASES COMPLETE"
echo " Finished: $(date)"
echo ""
echo " wandb projects:"
echo "   llm-reasoners-a1-lr-sweep-mps     → Phase 1: LR sweep"
echo "   llm-reasoners-a1-ablations-mps    → Phase 2-3: baseline + ablations"
echo "   llm-reasoners-a1-batch-sweep-mps  → Phase 4: batch size sweep"
echo ""
echo " Reminder: target val loss on MPS config is ≤ 2.00 (not 1.45)"
echo "============================================================"
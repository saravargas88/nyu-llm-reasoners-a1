#!/bin/bash
# run_experiments.sh
# Usage:
#   ./student/run_experiments.sh lr_sweep      ← run tonight (~5 hrs)
#   ./student/run_experiments.sh ablations     ← run tomorrow after checking wandb
#   ./student/run_experiments.sh batch_sweep   ← optional, skip if short on time

BASE="uv run python -m student.train"

MODEL="--vocab_size 10000 --context_length 256 --d_model 512 --d_ff 1344 --num_layers 4 --num_heads 16 --theta 10000"
TRAIN_CFG="--total_steps 15000 --warmup_steps 1000 --batch_size 64   --weight_decay 0.1 --beta1 0.9 --beta2 0.95 --grad_clip_max_l2_norm 1.0"

SECTION=${1:-"lr_sweep"}

# ─────────────────────────────────────────────────────────────────────────────
# LR sweep — 8 runs, ~5 hours overnight
# Coarse sweep (6) + 2 divergent runs for "edge of stability" deliverable
# ─────────────────────────────────────────────────────────────────────────────
run_lr_sweep() {
    echo "=== LR sweep (efficient edge-of-stability search) ==="

    for LR in 1e-3 1e-1; do
        echo "--- starting lr=${LR} ---"
        $BASE \
            $MODEL $TRAIN_CFG \
            --lr $LR \
            --lr_min $(uv run python3 -c "print($LR / 10)") \
            --run_name "lr_sweep__lr=${LR}" \
            --wandb_project "hpc-2-llm-a1-lr-sweep"
    done

    echo "=== LR sweep done — check wandb for best LR and divergence ==="
}
# ─────────────────────────────────────────────────────────────────────────────
# Ablations — 6 runs, ~3.5 hours
# Run tomorrow after finding best LR from wandb
# !! UPDATE BEST_LR BELOW BEFORE RUNNING !!
# ─────────────────────────────────────────────────────────────────────────────
run_ablations() {
    echo "=== Ablations ==="

    BEST_LR=3e-4    # ← UPDATE THIS after checking wandb from lr_sweep
    BEST_LR_MIN=$(uv run python3 -c "print($BEST_LR / 10)")

    ABLATION_CFG="--lr $BEST_LR --lr_min $BEST_LR_MIN $TRAIN_CFG"

    # baseline
    $BASE --train_path "$TRAIN_PATH" --val_path "$VAL_PATH" \
        $MODEL $ABLATION_CFG \
        --run_name "ablation__baseline"

    # remove all RMSNorm at best LR
    $BASE --train_path "$TRAIN_PATH" --val_path "$VAL_PATH" \
        $MODEL $ABLATION_CFG \
        --no_rmsnorm \
        --run_name "ablation__no_rmsnorm"

    # remove RMSNorm with lower LR to try to stabilise
    $BASE --train_path "$TRAIN_PATH" --val_path "$VAL_PATH" \
        $MODEL $TRAIN_CFG \
        --lr 1e-4 --lr_min 1e-5 \
        --no_rmsnorm \
        --run_name "ablation__no_rmsnorm_lowlr"

    # post-norm instead of pre-norm
    $BASE --train_path "$TRAIN_PATH" --val_path "$VAL_PATH" \
        $MODEL $ABLATION_CFG \
        --post_norm \
        --run_name "ablation__post_norm"

    # NoPE: no positional embeddings
    $BASE --train_path "$TRAIN_PATH" --val_path "$VAL_PATH" \
        $MODEL $ABLATION_CFG \
        --no_rope \
        --run_name "ablation__no_rope"

    # SiLU instead of SwiGLU
    $BASE --train_path "$TRAIN_PATH" --val_path "$VAL_PATH" \
        $MODEL $ABLATION_CFG \
        --use_silu \
        --run_name "ablation__silu"

    echo "=== Ablations done ==="
}

# ─────────────────────────────────────────────────────────────────────────────
# Batch sweep — optional, only 1 point, skip if short on time
# ─────────────────────────────────────────────────────────────────────────────
run_batch_sweep() {
    echo "=== Batch sweep (6 runs, ~3.5 hrs on MPS) ==="

    BEST_LR=3e-4    # ← UPDATE THIS after checking wandb from lr_sweep

    for BS in 1 8 32 64 128 256; do
        STEPS=$(uv run python3 -c "print(int(40_960_000 / ($BS * 256)))")
        WARMUP=$(uv run python3 -c "print(max(100, int($STEPS * 0.04)))")
        $BASE \
            --train_path "$TRAIN_PATH" --val_path "$VAL_PATH" \
            $MODEL \
            --lr $BEST_LR \
            --lr_min $(uv run python3 -c "print($BEST_LR / 10)") \
            --batch_size $BS --total_steps $STEPS --warmup_steps $WARMUP \
            --weight_decay 0.1 --beta1 0.9 --beta2 0.999 --grad_clip_max_l2_norm 1.0 \
            --run_name "batch_sweep__bs=${BS}"
    done

    echo "=== Batch sweep done ==="
}

case $SECTION in
    lr_sweep)    run_lr_sweep ;;
    ablations)   run_ablations ;;
    batch_sweep) run_batch_sweep ;;
    all)
        run_lr_sweep
        run_ablations
        ;;
    *)
        echo "Unknown section: $SECTION"
        echo "Usage: $0 [lr_sweep|ablations|batch_sweep|all]"
        exit 1
        ;;
esac

echo "=== All runs complete — check wandb.ai for curves ==="
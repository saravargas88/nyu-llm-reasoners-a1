#!/bin/bash
# run_experiments.sh
# Run all experiments for Section 7 systematically.
# Each experiment is one call to train.py with a descriptive --run_name.
# W&B logs everything — just overlay curves in the dashboard.
#
# Usage:
#   chmod +x run_experiments.sh
#   ./run_experiments.sh
#
# To run just one section:
#   ./run_experiments.sh lr_sweep
#   ./run_experiments.sh batch_sweep
#   ./run_experiments.sh ablations

BASE="python train.py"

# paths — edit once here, all runs pick them up
TRAIN="--train_path /Users/sara/Desktop/SPRING2026/LLM\ Reasoners/nyu-llm-reasoners-a1/data/results/train_tokens.npy"
VAL="--val_path /Users/sara/Desktop/SPRING2026/LLM\ Reasoners/nyu-llm-reasoners-a1/data/valid_tokens.npy"

# base model config — fixed across all experiments
MODEL="--vocab_size 10000 --context_length 256 --d_model 512 --d_ff 1344 --num_layers 4 --num_heads 16 --rope_theta 10000"

# training config — fixed across all experiments  
TRAIN_CFG="--total_steps 5000 --warmup_steps 200 --batch_size 64 --weight_decay 0.1 --beta1 0.9 --beta2 0.999 --grad_clip_max_l2_norm 1.0"

SECTION=${1:-"all"}

# ─────────────────────────────────────────────────────────────────────────────
# Problem: learning_rate — sweep then find edge of stability
# ─────────────────────────────────────────────────────────────────────────────

run_lr_sweep() {
    echo "=== LR sweep ==="

    # coarse sweep first — powers of 10
    for LR in 1e-5 1e-4 3e-4 1e-3 3e-3 1e-2; do
        $BASE $TRAIN $VAL $MODEL $TRAIN_CFG \
            --lr $LR --lr_min $(python3 -c "print($LR / 10)") \
            --run_name "lr_sweep__lr=${LR}"
    done

    # fine sweep around the best region (adjust after coarse sweep)
    for LR in 2e-4 5e-4 7e-4; do
        $BASE $TRAIN $VAL $MODEL $TRAIN_CFG \
            --lr $LR --lr_min $(python3 -c "print($LR / 10)") \
            --run_name "lr_fine__lr=${LR}"
    done

    # edge of stability — increasing LR until divergence
    for LR in 1e-2 3e-2 1e-1 3e-1; do
        $BASE $TRAIN $VAL $MODEL $TRAIN_CFG \
            --lr $LR --lr_min $(python3 -c "print($LR / 10)") \
            --run_name "lr_stability__lr=${LR}"
    done
}

# ─────────────────────────────────────────────────────────────────────────────
# Problem: batch_size_experiment
# ─────────────────────────────────────────────────────────────────────────────

run_batch_sweep() {
    echo "=== Batch size sweep ==="

    # use your best lr from the sweep above
    BEST_LR=3e-4

    for BS in 1 8 32 64 128 256; do
        # keep total tokens processed constant: steps = 327680000 / (bs * ctx)
        STEPS=$(python3 -c "print(int(327_680_000 / ($BS * 256)))")
        WARMUP=$(python3 -c "print(max(100, int($STEPS * 0.04)))")

        $BASE $TRAIN $VAL $MODEL \
            --lr $BEST_LR --lr_min $(python3 -c "print($BEST_LR / 10)") \
            --batch_size $BS --total_steps $STEPS --warmup_steps $WARMUP \
            --weight_decay 0.1 --beta1 0.9 --beta2 0.999 --grad_clip_max_l2_norm 1.0 \
            --run_name "batch_sweep__bs=${BS}"
    done
}

# ─────────────────────────────────────────────────────────────────────────────
# Problem: ablations (Section 7.3)
# Each flag turns off one component — compare against the baseline
# ─────────────────────────────────────────────────────────────────────────────

run_ablations() {
    echo "=== Ablations ==="

    BEST_LR=3e-4
    ABLATION_CFG="--lr $BEST_LR --lr_min 3e-5 $TRAIN_CFG"

    # baseline — full model, best lr
    $BASE $TRAIN $VAL $MODEL $ABLATION_CFG \
        --run_name "ablation__baseline"

    # layer_norm_ablation: remove RMSNorm
    $BASE $TRAIN $VAL $MODEL $ABLATION_CFG \
        --no_rmsnorm \
        --run_name "ablation__no_rmsnorm"

    # layer_norm_ablation: lower lr to try to stabilise without norm
    $BASE $TRAIN $VAL $MODEL $ABLATION_CFG \
        --no_rmsnorm --lr 1e-4 --lr_min 1e-5 \
        --run_name "ablation__no_rmsnorm_lowlr"

    # pre_norm_ablation: switch to post-norm
    $BASE $TRAIN $VAL $MODEL $ABLATION_CFG \
        --post_norm \
        --run_name "ablation__post_norm"

    # no_pos_emb: NoPE
    $BASE $TRAIN $VAL $MODEL $ABLATION_CFG \
        --no_rope \
        --run_name "ablation__no_rope"

    # swiglu_ablation: plain SiLU (d_ff auto-set to 4*d_model inside train.py)
    $BASE $TRAIN $VAL $MODEL $ABLATION_CFG \
        --use_silu \
        --run_name "ablation__silu"
}

# ─────────────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────────────

case $SECTION in
    lr_sweep)   run_lr_sweep ;;
    batch_sweep) run_batch_sweep ;;
    ablations)  run_ablations ;;
    all)
        run_lr_sweep
        run_batch_sweep
        run_ablations
        ;;
    *)
        echo "Unknown section: $SECTION"
        echo "Usage: $0 [lr_sweep|batch_sweep|ablations|all]"
        exit 1
        ;;
esac

echo "=== All runs complete — check wandb.ai for curves ==="

BASE="python train.py"

# FIX 7: use quoted paths (spaces in path break bash without quotes)
TRAIN_PATH="/Users/sara/Desktop/SPRING2026/LLM Reasoners/nyu-llm-reasoners-a1/data/results/train_tokens.npy"
VAL_PATH="/Users/sara/Desktop/SPRING2026/LLM Reasoners/nyu-llm-reasoners-a1/data/valid_tokens.npy"

# FIX 7: arg names now match train.py arg_parse()
#         --num_layers / --num_heads / --theta  (not --layers/--heads/--rope_theta)
MODEL="--vocab_size 10000 --context_length 256 --d_model 512 --d_ff 1344 --num_layers 4 --num_heads 16 --theta 10000"
TRAIN_CFG="--total_steps 5000 --warmup_steps 200 --batch_size 64 --weight_decay 0.1 --beta1 0.9 --beta2 0.999 --grad_clip_max_l2_norm 1.0"

SECTION=${1:-"all"}

run_lr_sweep() {
    echo "=== LR sweep ==="

    for LR in 1e-5 1e-4 3e-4 1e-3 3e-3 1e-2; do
        $BASE \
            --train_path "$TRAIN_PATH" --val_path "$VAL_PATH" \
            $MODEL $TRAIN_CFG \
            --lr $LR --lr_min $(python3 -c "print($LR / 10)") \
            --run_name "lr_sweep__lr=${LR}"
    done

    for LR in 2e-4 5e-4 7e-4; do
        $BASE \
            --train_path "$TRAIN_PATH" --val_path "$VAL_PATH" \
            $MODEL $TRAIN_CFG \
            --lr $LR --lr_min $(python3 -c "print($LR / 10)") \
            --run_name "lr_fine__lr=${LR}"
    done

    for LR in 1e-2 3e-2 1e-1 3e-1; do
        $BASE \
            --train_path "$TRAIN_PATH" --val_path "$VAL_PATH" \
            $MODEL $TRAIN_CFG \
            --lr $LR --lr_min $(python3 -c "print($LR / 10)") \
            --run_name "lr_stability__lr=${LR}"
    done
}

run_batch_sweep() {
    echo "=== Batch size sweep ==="

    BEST_LR=3e-4   # update after lr_sweep

    for BS in 1 8 32 64 128 256; do
        STEPS=$(python3 -c "print(int(327_680_000 / ($BS * 256)))")
        WARMUP=$(python3 -c "print(max(100, int($STEPS * 0.04)))")
        $BASE \
            --train_path "$TRAIN_PATH" --val_path "$VAL_PATH" \
            $MODEL \
            --lr $BEST_LR --lr_min $(python3 -c "print($BEST_LR / 10)") \
            --batch_size $BS --total_steps $STEPS --warmup_steps $WARMUP \
            --weight_decay 0.1 --beta1 0.9 --beta2 0.999 --grad_clip_max_l2_norm 1.0 \
            --run_name "batch_sweep__bs=${BS}"
    done
}

run_ablations() {
    echo "=== Ablations ==="

    BEST_LR=3e-4   # update after lr_sweep
    ABLATION_CFG="--lr $BEST_LR --lr_min 3e-5 $TRAIN_CFG"

    $BASE --train_path "$TRAIN_PATH" --val_path "$VAL_PATH" \
        $MODEL $ABLATION_CFG --run_name "ablation__baseline"

    $BASE --train_path "$TRAIN_PATH" --val_path "$VAL_PATH" \
        $MODEL $ABLATION_CFG --no_rmsnorm --run_name "ablation__no_rmsnorm"

    $BASE --train_path "$TRAIN_PATH" --val_path "$VAL_PATH" \
        $MODEL $ABLATION_CFG --no_rmsnorm --lr 1e-4 --lr_min 1e-5 \
        --run_name "ablation__no_rmsnorm_lowlr"

    $BASE --train_path "$TRAIN_PATH" --val_path "$VAL_PATH" \
        $MODEL $ABLATION_CFG --post_norm --run_name "ablation__post_norm"

    $BASE --train_path "$TRAIN_PATH" --val_path "$VAL_PATH" \
        $MODEL $ABLATION_CFG --no_rope --run_name "ablation__no_rope"

    $BASE --train_path "$TRAIN_PATH" --val_path "$VAL_PATH" \
        $MODEL $ABLATION_CFG --use_silu --run_name "ablation__silu"
}

case $SECTION in
    lr_sweep)    run_lr_sweep ;;
    batch_sweep) run_batch_sweep ;;
    ablations)   run_ablations ;;
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

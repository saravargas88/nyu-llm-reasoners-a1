#!/bin/bash
#SBATCH --job-name=lr_5e-2_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=compute
#SBATCH --output=logs/lr_5e-2.out

LR=1.5e-2

uv run python -m student.train \
  --train_path /root/Projects/train_tokens.npy \
  --val_path /root/Projects/valid_tokens.npy \
  --checkpoint_dir /scratch/ql2221/checkpoints_h200 \
  --checkpoint /scratch/ql2221/checkpoints_h200/lr_1.5e-2_shock/lr_1e-02/step_002500.pt \
  --device cuda \
  --batch_size 64 \
  --total_steps 3000 \
  --warmup_steps 0 \
  --lr $LR \
  --lr_min $LR \
  --weight_decay 0.1 \
  --beta1 0.9 \
  --beta2 0.999 \
  --grad_clip_max_l2_norm 1.0 \
  --wandb_project lr_divergence_test \
  --run_name lr_1.5e-2_shock
  

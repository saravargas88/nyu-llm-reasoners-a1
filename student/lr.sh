#!/bin/bash

#--checkpoint /scratch/ql2221/checkpoints_h200/lr_1.5e-2_shock/lr_1e-02/step_009999.pt \

LR=1e-3

#note i messed up the labeling the LR are all e-3
uv run python -m student.train \
  --train_path /root/Projects/train_tokens.npy \
  --val_path /root/Projects/valid_tokens.npy \
  --checkpoint_dir /scratch/ql2221/checkpoints_h200 \
  --device cuda \
  --batch_size 64 \
  --total_steps 50000 \
  --warmup_steps 0 \
  --lr $LR \
  --lr_min 1e-5 \
  --weight_decay 0.1 \
  --beta1 0.9 \
  --beta2 0.999 \
  --grad_clip_max_l2_norm 1.0 \
  --wandb_project lr_divergence_test \
  --run_name lr_1e-3_shock_lowerd_lrmin
  

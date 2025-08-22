#!/bin/bash

# 指定使用哪个 GPU（比如使用第 1 张卡，编号为 0）
export CUDA_VISIBLE_DEVICES=0

# 启动 Python 脚本，传入自定义参数
python main.py \
    --data_dir .datasets/teeth3ds \
    --num_points 16000 \
    --sample_points 16000 \
    --sample_views 4 \
    --batch_size 2 \
    --epochs 100 \
    --lr 1e-3 \
    --save_dir exp/baseline_biou_ptv3 \
    --eval_epoch_step 10 \
    --device cuda:0 \
    --num_workers 4 \
    --train_test_split 1 \
    --train \
    --use_wandb \

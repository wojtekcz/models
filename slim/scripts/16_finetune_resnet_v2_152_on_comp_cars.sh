#!/usr/bin/env bash

export TF_CPP_MIN_LOG_LEVEL=3

DATASET_DIR=/workspace/data/comp_cars
TRAIN_DIR=/workspace/train_logs/16_comp_cars_resnet_v2_152
CHECKPOINT_PATH=/workspace/checkpoints/resnet_v2_152.ckpt
DATASET_NAME=comp_cars
MODEL_NAME=resnet_v2_152

python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=${DATASET_NAME} \
    --dataset_split_name=train \
    --model_name=${MODEL_NAME} \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_exclude_scopes=resnet_v2_152/logits \
    --batch_size=48 \
    --save_summaries_secs=60 \
    --log_every_n_steps=100

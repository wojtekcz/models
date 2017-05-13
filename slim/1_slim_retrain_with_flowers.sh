#!/usr/bin/env bash

export TF_CPP_MIN_LOG_LEVEL=3

DATASET_DIR=/workspace/choose-network-architecture/data/flowers3
TRAIN_DIR=/workspace/choose-network-architecture/train_logs/1_flowers3
CHECKPOINT_PATH=/workspace/choose-network-architecture/checkpoints/inception_v3.ckpt
DATASET_NAME=flowers
MODEL_NAME=inception_v3

python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=${DATASET_NAME} \
    --dataset_split_name=train \
    --model_name=${MODEL_NAME} \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
    --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
    --save_summaries_secs=30

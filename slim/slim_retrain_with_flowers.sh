#!/usr/bin/env bash

export TF_CPP_MIN_LOG_LEVEL=3

DATASET_DIR=/workspace/data/flowers
TRAIN_DIR=/workspace/flowers-models/inception_v3
CHECKPOINT_PATH=/workspace/my_checkpoints/inception_v3.ckpt
DATASET_NAME=flowers
MODEL_NAME=inception_v3

cd /workspace/models/slim
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

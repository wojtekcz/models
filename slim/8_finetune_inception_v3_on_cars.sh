#!/usr/bin/env bash

export TF_CPP_MIN_LOG_LEVEL=3

DATASET_DIR=/workspace/data/car-reco3
TRAIN_DIR=/workspace/train_logs/8_car-reco3_inception_v3
CHECKPOINT_PATH=/workspace/checkpoints/inception_v3.ckpt
DATASET_NAME=cars
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
    --max_number_of_steps=20

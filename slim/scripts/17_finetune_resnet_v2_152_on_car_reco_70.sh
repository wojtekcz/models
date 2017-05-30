#!/usr/bin/env bash

export TF_CPP_MIN_LOG_LEVEL=3

DATASET_DIR=/workspace/data/car-reco3-70
TRAIN_DIR=/workspace/train_logs/17_car-reco3-70_resnet_v2_152
CHECKPOINT_PATH=/workspace/train_logs/16_comp_cars_resnet_v2_152
DATASET_NAME=car_reco3_70
MODEL_NAME=resnet_v2_152

python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=${DATASET_NAME} \
    --dataset_split_name=train \
    --model_name=${MODEL_NAME} \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_exclude_scopes=resnet_v2_152/logits \
    --trainable_scopes=resnet_v2_152/logits,resnet_v2_152/block3,resnet_v2_152/block4 \
    --save_summaries_secs=60 \
    --log_every_n_steps=100

#    --checkpoint_exclude_scopes=resnet_v2_152/logits,resnet_v2_152/block3,resnet_v2_152/block4  \

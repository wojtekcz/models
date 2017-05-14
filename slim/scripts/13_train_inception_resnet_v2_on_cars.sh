#!/usr/bin/env bash

export TF_CPP_MIN_LOG_LEVEL=3

DATASET_DIR=/workspace/data/car-reco3-70
TRAIN_DIR=/workspace/train_logs/13_car-reco3-70_inception_resnet_v2
DATASET_NAME=cars
MODEL_NAME=inception_resnet_v2

python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=${DATASET_NAME} \
    --dataset_split_name=train \
    --model_name=${MODEL_NAME} \
    --save_summaries_secs=60 \
    --save_summaries_secs=60 \
    --log_every_n_steps=100
#    --save_interval_secs=1800
#  --batch_size=32 \

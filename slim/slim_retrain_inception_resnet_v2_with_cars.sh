#!/usr/bin/env bash

export TF_CPP_MIN_LOG_LEVEL=3

DATASET_DIR=/workspace/choose-network-architecture/data/car-reco3
TRAIN_DIR=/workspace/choose-network-architecture/train_logs/4_car-reco3_inception_resnet_v2
CHECKPOINT_PATH=/workspace/choose-network-architecture/checkpoints/inception_resnet_v2_2016_08_30.ckpt
DATASET_NAME=cars
MODEL_NAME=inception_resnet_v2

python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=${DATASET_NAME} \
    --dataset_split_name=train \
    --model_name=${MODEL_NAME} \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
    --trainable_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
    --save_summaries_secs=60 \
    --log_every_n_steps=100
# InceptionV3/Logits
# InceptionV3/AuxLogits
# InceptionResnetV2/
# AuxLogits
# Logits
# --checkpoint_exclude_scopes=resnet_v1_50/logits \
#  --trainable_scopes=resnet_v1_50/logits \
#  --batch_size=32 \
#  --save_summaries_secs=60 \
#  --log_every_n_steps=100 \

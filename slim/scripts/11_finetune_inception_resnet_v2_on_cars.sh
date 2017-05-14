#!/usr/bin/env bash

export TF_CPP_MIN_LOG_LEVEL=3

DATASET_DIR=/workspace/data/car-reco3-70
TRAIN_DIR=/workspace/train_logs/11_car-reco3-70_inception_resnet_v2
CHECKPOINT_PATH=/workspace/checkpoints/inception_resnet_v2_2016_08_30.ckpt
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
#  --batch_size=32 \

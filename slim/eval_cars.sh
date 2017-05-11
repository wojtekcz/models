#!/usr/bin/env bash

CHECKPOINT_DIR=/Users/wcz/Beanflows/All_Beans/Machine_Learning/old_polish_cars/TF-slim-v5/inception_v3
DATASET_DIR=/Users/wcz/Beanflows/All_Beans/Machine_Learning/old_polish_cars/TF-slim-v5/data/cars-tfrecord

#model.ckpt-5810
#model.ckpt-6397
#model.ckpt-6982
#model.ckpt-7569
#model.ckpt-8152

CHECKPOINT_FILE=${CHECKPOINT_DIR}/model.ckpt-8152
EVAL_DIR=/Users/wcz/Beanflows/All_Beans/Machine_Learning/old_polish_cars/TF-slim-v5/eval

python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_FILE} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=cars \
    --dataset_split_name=validation \
    --model_name=inception_v3 \
    --eval_dir=${EVAL_DIR}
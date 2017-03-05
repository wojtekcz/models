#!/usr/bin/env bash

DATA_DIR=/workspace/data/cars

python download_and_convert_data.py \
    --dataset_name=cars \
    --dataset_dir="${DATA_DIR}"

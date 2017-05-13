#!/usr/bin/env bash

DATA_DIR=/Users/wcz/Beanflows/All_Beans/Machine_Learning/innovavant-Training-Car-Recognition-Model/choose-network-architecture/data/car-reco3

python download_and_convert_data.py \
    --dataset_name=cars \
    --dataset_dir="${DATA_DIR}"

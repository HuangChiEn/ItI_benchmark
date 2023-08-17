#!/usr/bin/env bash

set -x

NAME='ast_cloudy2sunnyweather'
TASK='AST'
DATA='cloudy2sunnyweather'
CROOT='/data1/dataset/WeatherGAN'
SROOT='/data1/dataset/WeatherGAN'
CKPTROOT='./checkpoints'
WORKER=4

# if you want to change the default setup, modify the following cfg file : 
# /data/joseph/wea_trfs_benchmark/TSIT/options/base_options.py

# we setup the run-time cmdline args
python train.py \
    --name $NAME \
    --task $TASK \
    --gpu_ids 0 \
    --checkpoints_dir $CKPTROOT \
    --batchSize 1 \
    --dataset_mode $DATA \
    --croot $CROOT \
    --sroot $SROOT \
    --nThreads $WORKER \
    --no_pairing_check \
    --no_instance \
    --gan_mode hinge \
    --num_upsampling_layers more \
    --alpha 1.0 \
    --display_freq 200 \
    --save_epoch_freq 20 \
    --niter 200 \
    --lambda_vgg 1 \
    --lambda_feat 1

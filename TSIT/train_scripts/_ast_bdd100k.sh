#!/usr/bin/env bash

set -x

NAME='ast_bdd100k'
TASK='AST'
DATA='bdd100k'
CROOT='/data1/dataset/bdd100k'
SROOT=$CROOT
CKPTROOT='./checkpoints'
WORKER=0

# if you want to change the default setup, modify the following cfg file : 
# /data/joseph/wea_trfs_benchmark/TSIT/options/base_options.py

# we setup the run-time cmdline args
python train.py \
    --name $NAME \
    --task $TASK \
    --gpu_ids 0 \
    --checkpoints_dir $CKPTROOT \
    --batchSize 8 \
    --dataset_mode $DATA \
    --croot $CROOT \
    --sroot $SROOT \
    --c_domain clear \
    --s_domain rainy \
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

# # no-pair chk to speedup ! # note this arg, we don't have instance! 
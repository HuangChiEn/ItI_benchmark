#!/usr/bin/env bash
#  The parameters configured in this script mainly
#  follow "TSIT/train_scripts/mmis_sunny2diffweathers.sh".

DEBUG=False
[ $DEBUG == 'True' ] && set -x

NAME='mmis_clear2diffweathers'
TASK='MMIS'
DATA='clear2diffweathers'
CROOT='/data1/dataset/bdd100k'
SROOT=$CROOT
CKPTROOT='./checkpoints'
WORKER=4

python train.py \
    --name $NAME \
    --task $TASK \
    --gpu_ids 0 \
    --checkpoints_dir $CKPTROOT \
    --batchSize 32 \
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
    --save_epoch_freq 5 \
    --niter 10 \
    --lambda_vgg 1 \
    --lambda_feat 1
    # original niter == 20 ep, niter_decay == 0
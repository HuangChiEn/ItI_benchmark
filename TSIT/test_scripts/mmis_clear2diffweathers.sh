#!/usr/bin/env bash

DEBUG=False
[ $DEBUG == 'True' ] && set -x

NAME='mmis_clear2diffweathers'
TASK='MMIS'
DATA='clear2diffweathers'
CROOT='/data1/dataset/bdd100k'
SROOT=$CROOT
CKPTROOT='./checkpoints'
WORKER=4

RESROOT='./results'
EPOCH='latest'
MODE=${1:-'all'}

python test.py \
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
    --num_upsampling_layers more \
    --alpha 1.0 \
    --results_dir $RESROOT \
    --which_epoch $EPOCH \
    --show_input \
    --test_mode $MODE

#!/usr/bin/env bash
#  The parameters configured in this script mainly
#  follow "TSIT/train_scripts/mmis_sunny2diffweathers.sh".
#  'mmis' is exactly the task we want, but the meta-data 
#  what mmis required only contains fewer weather condition.
#  So, we use ast mode to simulate mmis task in this script!

DEBUG=False
[ "$DEBUG" == 'True' ] && set -x  # set -x print all info to console


NAME='ast_bdd100k'
TASK='AST'
DATA='bdd100k'

CROOT='/data1/dataset/bdd100k'
SROOT=$CROOT   # different domains are placed at same folder 

CKPTROOT='./ckpt'
WORKER=4

src_lst=('clear' 'clear' 'clear')
tar_lst=('rainy' 'snowy' 'overcast')

for index in "${!src_lst[@]}" ; do
    # zip the src -> tar domain trfs!
    src=${src_lst[$index]} ; tar=${tar_lst[$index]}
    
    ckpt_str="${CKPTROOT}_${src}2${tar}"

    python train.py \
        --name $NAME \
        --task $TASK \
        --gpu_ids 0 \
        --checkpoints_dir $ckpt_str \
        --batchSize 32 \
        --dataset_mode $DATA \
        --croot $CROOT \
        --sroot $SROOT \
        --c_domain $src \
        --s_domain $tar \
        --nThreads $WORKER \
        --no_pairing_check \
        --no_instance \
        --gan_mode hinge \
        --num_upsampling_layers more \
        --alpha 1.0 \
        --display_freq 200 \
        --save_epoch_freq 4 \
        --niter 20 \
        --lambda_vgg 1 \
        --lambda_feat 1
done


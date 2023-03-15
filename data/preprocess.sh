#!/bin/bash

declare -a tasks=( "qnli" "hans") # define tasks, eg. qnli, hans

for ((i=0; i<${#tasks[@]}; i++));
do  
    python3 preprocess.py   \
    --model_name_or_path bert-base-uncased  \
    --task_name ${tasks[i]}   \
    --max_length 256   \
    --per_device_train_batch_size 16   \
    --learning_rate 2e-5   \
    --num_train_epochs 1   \
    --output_dir "datasets/${tasks[i]}/" \
    --do_noniid True \
    --alpha 1000000 \
done
#!/bin/bash

declare -a clients=(2 4 8 16 32) 
declare -a seeds=(0 42 123) 
task=qnli

for ((i=0; i<${#clients[@]}; i++));
do 
    for ((j=0; j<${#seeds[@]}; j++));
    do
        python3 fed.py \
            --model_name_or_path bert-base-uncased \
            --task_name  ${task} \
            --fed_dir_data "datasets/" \
            --checkpointing_steps=epoch \
            --output_dir "results/${task}_${clients[i]}/${seeds[j]}/" \
            --num_train_epochs 1 \
            --num_rounds 10 \
            --do_noniid True \
            --alpha 1000 \
            --do_freeze True \
            --num_clients ${clients[i]} \
            --seed ${seeds[j]} \           
    done
done
    
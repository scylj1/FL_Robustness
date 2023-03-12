#!/bin/bash
#SBATCH -A LANE-SL3-GPU
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 7

# source your conda environment 
source /home/lj408/miniconda3/bin/activate fdabert

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
    --output_dir "/rds/user/lj408/hpc-work/datasets/noniid/${tasks[i]}/" \
    --do_noniid True \
    --alpha 1000 \
    --cache_dir "/rds/user/lj408/hpc-work/cache/fed/" 

done
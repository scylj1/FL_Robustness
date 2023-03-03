#!/bin/bash
#SBATCH --cpus-per-task 7 # cpu resources (usually 7 cpu cores per GPU)
#SBATCH --gres=gpu:rtx2080:1 # gpu resources ## use :gtx1080: or :rtx2080: or :v100: or :a40: (you can ask for more than 1 gpu if you want)
#SBATCH --job-name=q2 # a name just for you to identify your job easily

# source your conda environment (which should live in Aoraki)
source /nfs-share/lj408/miniconda3/bin/activate fdabert

srun python3 fed.py \
    --model_name_or_path "bert-base-uncased" \
    --fed_dir_data 'glue_data/' \
    --per_device_train_batch_size 8 \
    --checkpointing_steps=epoch \
    --output_dir "results/QNLI_2/" \
    --cache_dir "/nfs-share/lj408/FL_Robustness/cache/fed_mtl/" \
    --num_train_epochs 1 \
    --num_rounds 10 \
    --num_clients 2 \
    --seed 42 \
    
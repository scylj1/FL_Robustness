#!/bin/bash
#SBATCH --cpus-per-task 7 # cpu resources (usually 7 cpu cores per GPU)
#SBATCH --gres=gpu:rtx2080:1 # gpu resources ## use :gtx1080: or :rtx2080: or :v100: or :a40: (you can ask for more than 1 gpu if you want)
#SBATCH --job-name=fed # a name just for you to identify your job easily

# source your conda environment (which should live in Aoraki)
source /nfs-share/lj408/miniconda3/bin/activate fdabert

srun python preprocess.py   \
--model_name_or_path bert-base-uncased   \
--task_name hans   \
--max_length 256   \
--per_device_train_batch_size 16   \
--learning_rate 2e-5   \
--num_train_epochs 3   \
--output_dir HANS/ \
--cache_dir /nfs-share/lj408/FL_Robustness/cache/fed_mtl/
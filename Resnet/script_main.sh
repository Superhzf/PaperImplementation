#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_6000:1

module load ml-gpu
cd /work/LAS/cjquinn-lab/zefuh/COMS673/CS673/HW2/code
ml-gpu /work/LAS/cjquinn-lab/zefuh/COMS673/DL_env/bin/python main.py -seed 2022 -resnet_version 2 -resnet_size 3 -num_classes 10 -num_epochs 164 -batch 128 -lr 0.1 -> log_2022_v2

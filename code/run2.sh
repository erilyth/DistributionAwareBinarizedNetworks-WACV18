#!/bin/bash
#SBATCH -A research
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=48:00:00
#SBATCH --mincpus=2
#SBATCH --nodelist=gnode29
module add cuda/8.0
module add cudnn/7-cuda-8.0

srun bash clean.sh; CUDA_VISIBLE_DEVICES=1 python3 main.py --dataset='cifar10' --data_dir='/ssd_scratch/cvit/' --nClasses=10 --workers=2 --epochs=400 --batch-size=256 --testbatchsize=16 --learningratescheduler='decayschedular' --decayinterval=50 --decaylevel=2 --optimType='adam' --nesterov --maxlr=0.002 --minlr=0.00005 --weightDecay=0 --binaryWeight --binStart=2 --binEnd=7 --model_def='bincifar' --inpsize=225 --name='cifar10_bincifar_dualbin' | tee "textlogs/cifar10_bincifar_dualbin.txt"

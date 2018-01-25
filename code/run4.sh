#!/bin/bash
#SBATCH -A vishal.b
#SBATCH -n 1
#SBATCH --gres=gpu:3
#SBATCH --mem-per-cpu=2048
#SBATCH --time=48:00:00
#SBATCH --mincpus=24
#SBATCH --nodelist=gnode32
module add cuda/8.0
module add cudnn/7-cuda-8.0


srun bash clean.sh; CUDA_VISIBLE_DEVICES=0 python3 main.py --dataset='tuberlin' --data_dir='/ssd_scratch/cvit/' --nClasses=250 --workers=4 --epochs=600 --batch-size=96 --testbatchsize=4 --learningratescheduler='decayschedular' --decayinterval=50 --decaylevel=2 --optimType='adam' --nesterov --tenCrop --maxlr=0.002 --minlr=0.0001 --weightDecay=0 --binaryWeight --binStart=2 --binEnd=100 --model_def='googlenetfbin' --inpsize=224 --name='tuberlin_googlenetfbin_dualbin_oldactiv' | tee "textlogs/tuberlin_googlenetfbin_dualbin_oldactiv.txt" &

srun bash clean.sh; CUDA_VISIBLE_DEVICES=1 python3 main.py --dataset='tuberlin' --data_dir='/ssd_scratch/cvit/' --nClasses=250 --workers=4 --epochs=600 --batch-size=96 --testbatchsize=4 --learningratescheduler='decayschedular' --decayinterval=50 --decaylevel=2 --optimType='adam' --nesterov --tenCrop --maxlr=0.001 --minlr=0.00002 --weightDecay=0 --binaryWeight --binStart=2 --binEnd=100 --model_def='googlenetwbin' --inpsize=224 --name='tuberlin_googlenetwbin_dualbin' | tee "textlogs/tuberlin_googlenetwbin_dualbin.txt" &

srun bash clean.sh; CUDA_VISIBLE_DEVICES=2 python3 main.py --dataset='sketchyrecognition' --data_dir='/ssd_scratch/cvit/' --nClasses=125 --workers=4 --epochs=400 --batch-size=96 --testbatchsize=4 --learningratescheduler='decayschedular' --decayinterval=50 --decaylevel=2 --optimType='adam' --nesterov --tenCrop --maxlr=0.002 --minlr=0.0001 --weightDecay=0 --binaryWeight --binStart=2 --binEnd=100 --model_def='googlenetfbin' --inpsize=224 --name='sketchy_googlenetfbin_dualbin_oldactiv' | tee "textlogs/sketchy_googlenetfbin_dualbin_oldactiv.txt"

#srun bash clean.sh; CUDA_VISIBLE_DEVICES=3 python3 main.py --dataset='sketchyrecognition' --data_dir='/ssd_scratch/cvit/' --nClasses=125 --workers=4 --epochs=400 --batch-size=128 --testbatchsize=8 --learningratescheduler='decayschedular' --decayinterval=50 --decaylevel=2 --optimType='adam' --nesterov --tenCrop --maxlr=0.001 --minlr=0.00002 --weightDecay=0 --binaryWeight --binStart=2 --binEnd=100 --model_def='googlenetwbin' --inpsize=224 --name='sketchy_googlenetwbin_dualbin' | tee "textlogs/sketchy_googlenetwbin_dualbin.txt"

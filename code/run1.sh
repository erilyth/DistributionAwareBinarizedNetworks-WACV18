#!/bin/bash
#SBATCH -A research
#SBATCH -n 1
#SBATCH --gres=gpu:3
#SBATCH --mem-per-cpu=2048
#SBATCH --time=48:00:00
#SBATCH --mincpus=24
#SBATCH --nodelist=gnode27
module add cuda/8.0
module add cudnn/7-cuda-8.0


srun bash clean.sh; CUDA_VISIBLE_DEVICES=0 python3 main.py --dataset='tuberlin' --data_dir='/ssd_scratch/cvit/' --nClasses=250 --workers=8 --epochs=600 --batch-size=512 --testbatchsize=16 --learningratescheduler='decayschedular' --decayinterval=50 --decaylevel=2 --optimType='adam' --nesterov --tenCrop --maxlr=0.0005 --minlr=0.00001 --weightDecay=0 --binaryWeight --binStart=2 --binEnd=7 --model_def='sketchanetwbin' --inpsize=225 --name='tuberlin_sketchanetwbin_dualbin' | tee "textlogs/tuberlin_sketchanetwbin_dualbin.txt" &

srun bash clean.sh; CUDA_VISIBLE_DEVICES=1  python3 main.py --dataset='tuberlin' --data_dir='/ssd_scratch/cvit/' --nClasses=250 --workers=8 --epochs=600 --batch-size=256 --testbatchsize=16 --learningratescheduler='decayschedular' --decayinterval=50 --decaylevel=2 --optimType='adam' --nesterov --tenCrop --maxlr=0.0005 --minlr=0.00001 --weightDecay=0 --binaryWeight --binStart=2 --binEnd=100 --model_def='resnetwbin18' --inpsize=224 --name='tuberlin_resnetwbin18_dualbin' | tee "textlogs/tuberlin_resnetwbin18_dualbin.txt" &

srun bash clean.sh; CUDA_VISIBLE_DEVICES=2  python3 main.py --dataset='tuberlin' --data_dir='/ssd_scratch/cvit/' --nClasses=250 --workers=8 --epochs=600 --batch-size=128 --testbatchsize=4 --learningratescheduler='decayschedular' --decayinterval=50 --decaylevel=2 --optimType='adam' --nesterov --tenCrop --maxlr=0.0005 --minlr=0.00001 --weightDecay=0 --binaryWeight --binStart=2 --binEnd=100 --model_def='squeezenetwbin' --inpsize=224 --name='tuberlin_squeezenetwbin_dualbin' | tee "textlogs/tuberlin_squeezenetwbin_dualbin.txt"


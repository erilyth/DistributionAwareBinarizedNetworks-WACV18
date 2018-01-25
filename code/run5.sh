#!/bin/bash
#SBATCH -A ameya.prabhu
#SBATCH -n 1
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=2048
#SBATCH --time=48:00:00
#SBATCH --mincpus=16
#SBATCH --nodelist=gnode26
module add cuda/8.0
module add cudnn/7-cuda-8.0

srun bash clean.sh; CUDA_VISIBLE_DEVICES=0 python3 main.py --dataset='sketchyrecognition' --data_dir='/ssd_scratch/cvit/' --nClasses=125 --workers=4 --epochs=400 --batch-size=256 --testbatchsize=16 --learningratescheduler='decayschedular' --decayinterval=50 --decaylevel=2 --optimType='adam' --nesterov --tenCrop --maxlr=0.0005 --minlr=0.00001 --weightDecay=0 --binaryWeight --binStart=2 --binEnd=7 --model_def='sketchanetfbin' --inpsize=225 --name='sketchy_sketchanetfbin_dualbin_oldactiv' | tee "textlogs/sketchy_sketchanetfbin_dualbin_oldactiv.txt" &

srun bash clean.sh; CUDA_VISIBLE_DEVICES=1 python3 main.py --dataset='sketchyrecognition' --data_dir='/ssd_scratch/cvit/' --nClasses=125 --workers=4 --epochs=400 --batch-size=128 --testbatchsize=8 --learningratescheduler='decayschedular' --decayinterval=50 --decaylevel=2 --optimType='adam' --nesterov --tenCrop --maxlr=0.0005 --minlr=0.00001 --weightDecay=0 --binaryWeight --binStart=2 --binEnd=100 --model_def='resnetfbin18' --inpsize=224 --name='sketchy_resnetfbin18_dualbin_oldactiv' | tee "textlogs/sketchy_resnetfbin18_dualbin_oldactiv.txt" &

#srun bash clean.sh; CUDA_VISIBLE_DEVICES=2 python3 main.py --dataset='tuberlin' --data_dir='/ssd_scratch/cvit/' --nClasses=250 --workers=4 --epochs=600 --batch-size=128 --testbatchsize=8 --learningratescheduler='decayschedular' --decayinterval=50 --decaylevel=2 --optimType='adam' --nesterov --tenCrop --maxlr=0.0005 --minlr=0.00001 --weightDecay=0 --binaryWeight --binStart=2 --binEnd=100 --model_def='squeezenetfbin' --inpsize=224 --name='tuberlin_squeezenetfbin_dualbin_oldactiv' | tee "textlogs/tuberlin_squeezenetfbin_dualbin_oldactiv.txt"

#srun bash clean.sh; CUDA_VISIBLE_DEVICES=3 python3 main.py --dataset='cifar100' --data_dir='/ssd_scratch/cvit/' --nClasses=100 --workers=2 --epochs=600 --batch-size=256 --testbatchsize=16 --learningratescheduler='decayschedular' --decayinterval=50 --decaylevel=2 --optimType='adam' --nesterov --maxlr=0.001 --minlr=0.00005 --weightDecay=0 --binaryWeight --binStart=2 --binEnd=7 --model_def='bincifar' --inpsize=225 --name='cifar100_bincifar_dualbin' | tee "textlogs/cifar100_bincifar_dualbin.txt"

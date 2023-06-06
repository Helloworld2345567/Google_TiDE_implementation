#!/bin/bash
#SBATCH --array=70-71
#SBATCH -p rise # partition (queue)
#SBATCH --nodelist=pavia
#SBATCH -n 1 # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=6 # number of cores per task
#SBATCH --gres=gpu:2
#SBATCH -t 1-24:00 # time requested (D-HH:MM)
#LayerNorm True revIn False
pwd
hostname
date
echo starting job...
source ~/.bashrc

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

root=/root/data1/ym/google_tide/TiDE
cd ${root}

name=TiDE
print_tofile=False
datadir=${root}/data
cuda=True
dataset=ETTm1
epoch=100
batch_size=512
cuda=True
lr=8.4e-5
ckpt_path=/root/data1/ym/google_tide/TiDE/result/dataset_${dataset}
save_path=${ckpt_path}
drop_prob=0.5
#hidden_size=256
#num_encoder_layers=2
#num_decoder_layers=2
#decoder_output_dim=8
#temporal_decoder_hidden=128
mkdir -p ${ckpt_path}

cd src
pwd
#you can modify python train.py to python val.py to val
CUDA_VISIBLE_DEVICES=0 \
    python train.py \
    --name ${name} \
    --print-tofile ${print_tofile} \
    --ckpt_path ${ckpt_path} \
    --datadir ${datadir} \
    --dataset ${dataset} \
    --save_path ${save_path} \
    --epoch ${epoch} \
    --batch_size ${batch_size} \
    --cuda ${cuda} \
    --lr ${lr} \
    --drop_prob ${drop_prob} \
    --hidden_size=1024 \
    --num_encoder_layers=1 \
    --num_decoder_layers=1 \
    --decoder_output_dim=8 \
    --temporal_decoder_hidden=128 \
    --pred_len=96 \
    --lookback_len=720 \
    --layer_norm=True \
    --revin=False\

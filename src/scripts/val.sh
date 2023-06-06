#!/bin/bash
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
dataset=traffic
epoch=50
batch_size=1
cuda=True
lr=3.82e-4
ckpt_path=/root/data1/ym/google_tide/TiDE/result/dataset_${dataset}
save_path=${ckpt_path}
drop_prob=0.3

mkdir -p ${ckpt_path}

cd src
pwd
CUDA_VISIBLE_DEVICES=0 python val.py \
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
    --hidden_size=256 \
    --num_encoder_layers=1 \
    --num_decoder_layers=1 \
    --decoder_output_dim=16 \
    --temporal_decoder_hidden=64 \
    --pred_len=96 \
    --lookback_len=720 \
    --layer_norm=False \
    --revin=True\
    

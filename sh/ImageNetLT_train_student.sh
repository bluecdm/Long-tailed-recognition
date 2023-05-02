#!/bin/bash

b=1.0
g=1.0
t=0.07
k=16384
bk=3
n=1

torchrun \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:29500 \
main.py \
  --multiprocessing_distributed \
  --workers 16 \
  --dataset imagenet \
  --arch resnext50_32x4d \
  --arch_t resnext101_32x8d \
  --path_t ./outputs/imagenet/X101_32x8d_randncls_90epochs_lr0.1_ce_1/moco_ckpt.pth.tar \
  --feat_t 2048 \
  --data ~/DATA_ROOT/IMAGENET-1K/imagenet/ \
  --mark X50_k"$k"_bk"$bk"_lr0.05_g"$g"_b"$b"_tX101_"$n" \
  --batch_size 128 \
  --learning_rate 0.05 \
  --epochs 90 \
  --aug randcls_randclsstack \
  --normalize true \
  --moco_t "$t" \
  --moco_k "$k" \
  --base_k "$bk" \
  --beta "$b" \
  --gamma "$g"
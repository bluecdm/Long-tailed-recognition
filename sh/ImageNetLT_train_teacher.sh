#!/bin/bash

b=0.0
g=1.0
n=1

torchrun \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:29500 \
main.py \
  --multiprocessing_distributed \
  --workers 16 \
  --dataset imagenet \
  --arch resnext101_32x8d \
  --data ~/DATA_ROOT/IMAGENET-1K/imagenet/ \
  --mark X101_32x8d_randncls_90epochs_lr0.1_ce_"$n" \
  --batch_size 128 \
  --learning_rate 0.05 \
  --epochs 90 \
  --aug randcls_randclsstack \
  --normalize true \
  --beta "$b" \
  --gamma "$g"
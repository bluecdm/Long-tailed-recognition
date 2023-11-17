#!/bin/bash

python cifar_main.py \
  --workers 8 \
  --dataset cifar100 \
  --imb-factor 0.01 \
  --arch resnet32 \
  --arch_t resnet56 \
  --path_t ./outputs/cifar100/ce_CIFAR100_imb001_R56_lr0.1_0/moco_ckpt.pth.tar \
  --feat_t 64 \
  --mark CIFAR100_imb001_R32_tR56_k4096_bk2_lr0.1_TL_g1_a1_b1_0 \
  --batch-size 128 \
  --learning-rate 0.1 \
  --gamma 1.0 \
  --alpha 1.0 \
  --beta 1.0 \
  --epoch-multiplier 1
#!/bin/bash

python cifar_ce.py \
  --workers 8 \
  --dataset cifar100 \
  --arch resnet56 \
  --imb-factor 0.01 \
  --batch-size 128 \
  --learning-rate 0.1 \
  --mark ce_CIFAR100_imb001_R56_lr0.1_0 \
  --epoch-multiplier 1

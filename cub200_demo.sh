#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

python demo.py --data cub200 --model model/best_epoch.pth

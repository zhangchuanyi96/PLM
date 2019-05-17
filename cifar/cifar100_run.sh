export CUDA_VISIBLE_DEVICES=0

python main.py --dataset cifar100 --bs 128 --lr 0.001 --n_epoch 80

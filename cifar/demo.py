# -*- coding:utf-8 -*-
import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from data.cifar import CIFAR10
from model import CNN
import argparse, sys
import numpy as np
import datetime
import shutil

from loss_plm import peer_learning_loss

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type = str, help = 'cifar10', default = 'cifar10')
parser.add_argument('--model', type=str, required=True)
args = parser.parse_args()

# Seed
torch.manual_seed(1)
torch.cuda.manual_seed(1)

# Hyper Parameters
batch_size=128
noise_type = 'pairflip'
noise_rate = 0.45
drop_rate = noise_rate
result_dir = 'result/'
epoch_decay_start = 60

# load dataset
if args.dataset=='cifar10':
    input_channel=3
    num_classes=10
    train_dataset = CIFAR10(root='./data/',
                                download=True,  
                                train=True, 
                                transform=transforms.ToTensor(),
                                noise_type=noise_type,
                                noise_rate=noise_rate
                           )
    
    test_dataset = CIFAR10(root='./data/',
                                download=True,  
                                train=False, 
                                transform=transforms.ToTensor(),
                                noise_type=noise_type,
                                noise_rate=noise_rate
                          )

# Evaluate the Model
def evaluate(test_loader, model):
    model.eval()    # Change model to 'eval' mode.
    correct = 0
    total = 0
    for images, labels, _ in test_loader:
        images = images.cuda()
        logits = model(images)
        outputs = F.softmax(logits, dim=1)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred.cpu() == labels).sum()
 
    acc = 100*float(correct)/float(total)
    return acc


def main():
    # Data Loader (Input Pipeline)
    print('loading dataset...')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               num_workers=4,
                                               drop_last=True,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              num_workers=4,
                                              drop_last=True,
                                              shuffle=False)
    # Define models
    print('building model...')
    cnn = CNN(input_channel=input_channel, n_outputs=num_classes)
    cnn.cuda()
    cnn.load_state_dict(torch.load(args.model))

    # evaluate models with random weights
    test_acc=evaluate(test_loader, cnn)
    print('=========> Test Accuracy on the %s test images: %.4f %% <===========' % (len(test_dataset), test_acc))


if __name__=='__main__':
    main()

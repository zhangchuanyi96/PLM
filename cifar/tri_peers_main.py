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

from tri_peers_loss_plm import peer_learning_loss

parser = argparse.ArgumentParser()
parser.add_argument('--bs', type = int, default = 128)
parser.add_argument('--lr', type = float, default = 0.001)
parser.add_argument('--T_k', type = int, default = 10, help='how many epochs for linear drop rate, can be 5, 10, 15.')
parser.add_argument('--dataset', type = str, help = 'cifar10', default = 'cifar10')
parser.add_argument('--n_epoch', type=int, default=200)
parser.add_argument('--print_freq', type=int, default=50)


args = parser.parse_args()

# Seed
torch.manual_seed(1)
torch.cuda.manual_seed(1)

# Hyper Parameters
batch_size = args.bs
learning_rate = args.lr
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

noise_or_not = train_dataset.noise_or_not

# Adjust learning rate and betas for Adam Optimizer
mom1 = 0.9
mom2 = 0.1
alpha_plan = [learning_rate] * args.n_epoch
beta1_plan = [mom1] * args.n_epoch
for i in range(epoch_decay_start, args.n_epoch):
    alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - epoch_decay_start) * learning_rate
    beta1_plan[i] = mom2

def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]
        param_group['betas']=(beta1_plan[epoch], 0.999) # Only change beta1
        
# define drop rate schedule
rate_schedule = np.ones(args.n_epoch)*drop_rate
rate_schedule[:args.T_k] = np.linspace(0, drop_rate, args.T_k)
   
save_dir = result_dir + '/' + args.dataset

if not os.path.exists(save_dir):
    os.system('mkdir -p %s' % save_dir)

model_str=args.dataset+'_peerlearning_'+noise_type+'_'+str(noise_rate)

txtfile=save_dir+"/"+model_str+".txt"
nowTime=datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
if os.path.exists(txtfile):
    os.system('mv %s %s' % (txtfile, txtfile+".bak-%s" % nowTime))


def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# Train the Model
def train(train_loader,epoch, model1, optimizer1, model2, optimizer2, model3, optimizer3):
    # print('Training %s...' % model_str)
    pure_ratio_list=[]
    pure_ratio_1_list=[]
    pure_ratio_2_list=[]
    pure_ratio_3_list=[]
    
    train_total=0
    train_correct=0 
    train_total2=0
    train_correct2=0
    train_total3=0
    train_correct3=0 

    for i, (images, labels, indexes) in enumerate(train_loader):
        ind=indexes.cpu().numpy().transpose()
      
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        
        # Forward + Backward + Optimize
        logits1=model1(images)
        prec1, _ = accuracy(logits1, labels, topk=(1, 5))
        train_total+=1
        train_correct+=prec1

        logits2 = model2(images)
        prec2, _ = accuracy(logits2, labels, topk=(1, 5))
        train_total2+=1
        train_correct2+=prec2
        
        logits3 = model3(images)
        prec3, _ = accuracy(logits3, labels, topk=(1, 5))
        train_total3+=1
        train_correct3+=prec3

        loss_1, loss_2, loss_3, pure_ratio_1, pure_ratio_2, pure_ratio_3 = peer_learning_loss(logits1, logits2, logits3, labels, rate_schedule[epoch], ind, noise_or_not)
        pure_ratio_1_list.append(100*pure_ratio_1)
        pure_ratio_2_list.append(100*pure_ratio_2)
        pure_ratio_3_list.append(100*pure_ratio_3)

        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()
        optimizer2.zero_grad()
        loss_2.backward()
        optimizer2.step()
        optimizer3.zero_grad()
        loss_3.backward()
        optimizer3.step()
        if (i+1) % args.print_freq == 0:
            print('Epoch [%d/%d], Iter [%03d/%03d] Training Accuracy1: %.4F, Training Accuracy2: %.4f, Training Accuracy3: %.4f, Loss1: %.4f, Loss2: %.4f, Loss3: %.4f, Pure Ratio1: %.4f, Pure Ratio2 %.4f, Pure Ratio3 %.4f' 
                  %(epoch+1, args.n_epoch, i+1, len(train_dataset)//batch_size, prec1, prec2, prec3, loss_1.item(), loss_2.item(), loss_3.item(), np.sum(pure_ratio_1_list)/len(pure_ratio_1_list), np.sum(pure_ratio_2_list)/len(pure_ratio_2_list), np.sum(pure_ratio_3_list)/len(pure_ratio_3_list)))

    train_acc1=float(train_correct)/float(train_total)
    train_acc2=float(train_correct2)/float(train_total2)
    train_acc3=float(train_correct3)/float(train_total3)
    return train_acc1, train_acc2, train_acc3, pure_ratio_1_list, pure_ratio_2_list, pure_ratio_3_list

# Evaluate the Model
def evaluate(test_loader, model1, model2, model3):
    print('Evaluating %s...' % model_str)
    model1.eval()    # Change model to 'eval' mode.
    correct1 = 0
    total1 = 0
    for images, labels, _ in test_loader:
        images = Variable(images).cuda()
        logits1 = model1(images)
        outputs1 = F.softmax(logits1, dim=1)
        _, pred1 = torch.max(outputs1.data, 1)
        total1 += labels.size(0)
        correct1 += (pred1.cpu() == labels).sum()

    model2.eval()    # Change model to 'eval' mode 
    correct2 = 0
    total2 = 0
    for images, labels, _ in test_loader:
        images = Variable(images).cuda()
        logits2 = model2(images)
        outputs2 = F.softmax(logits2, dim=1)
        _, pred2 = torch.max(outputs2.data, 1)
        total2 += labels.size(0)
        correct2 += (pred2.cpu() == labels).sum()

    model3.eval()    # Change model to 'eval' mode 
    correct3 = 0
    total3 = 0
    for images, labels, _ in test_loader:
        images = Variable(images).cuda()
        logits3 = model3(images)
        outputs3 = F.softmax(logits3, dim=1)
        _, pred3 = torch.max(outputs3.data, 1)
        total3 += labels.size(0)
        correct3 += (pred3.cpu() == labels).sum()
 
    acc1 = 100*float(correct1)/float(total1)
    acc2 = 100*float(correct2)/float(total2)
    acc3 = 100*float(correct3)/float(total3)
    return acc1, acc2, acc3


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
    cnn1 = CNN(input_channel=input_channel, n_outputs=num_classes)
    cnn1.cuda()
    print(cnn1.parameters)
    optimizer1 = torch.optim.Adam(cnn1.parameters(), lr=learning_rate)
    
    cnn2 = CNN(input_channel=input_channel, n_outputs=num_classes)
    cnn2.cuda()
    print(cnn2.parameters)
    optimizer2 = torch.optim.Adam(cnn2.parameters(), lr=learning_rate)

    cnn3 = CNN(input_channel=input_channel, n_outputs=num_classes)
    cnn3.cuda()
    print(cnn3.parameters)
    optimizer3 = torch.optim.Adam(cnn3.parameters(), lr=learning_rate)

    mean_pure_ratio1=0
    mean_pure_ratio2=0
    mean_pure_ratio3=0

    with open(txtfile, "a") as myfile:
        myfile.write('epoch: train_acc1 train_acc2 train_acc3 test_acc1 test_acc2 test_acc3 pure_ratio1 pure_ratio2\n')

    epoch=0
    train_acc1=0
    train_acc2=0
    train_acc3=0
    # evaluate models with random weights
    test_acc1, test_acc2, test_acc3 = evaluate(test_loader, cnn1, cnn2, cnn3)
    print('Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f %% Model3 %.4f %% Pure Ratio1 %.4f %% Pure Ratio2 %.4f %% Pure Ratio3 %.4f %%' % (epoch+1, args.n_epoch, len(test_dataset), test_acc1, test_acc2, test_acc3, mean_pure_ratio1, mean_pure_ratio2, mean_pure_ratio3))
    # save results
    with open(txtfile, "a") as myfile:
        output = '{:05d}: {:10.4f} {:10.4f} {:10.4f} {:9.4f} {:9.4f} {:9.4f} {:11.4f} {:11.4f} {:11.4f}\n'.format(epoch, train_acc1, train_acc2, train_acc3, test_acc1, test_acc2, test_acc3, mean_pure_ratio1, mean_pure_ratio2, mean_pure_ratio3)
        myfile.write(output)
        # myfile.write(str(int(epoch)) + ': '  + str(train_acc1) +' '  + str(train_acc2) +' '  + str(test_acc1) + " " + str(test_acc2) + ' '  + str(mean_pure_ratio1) + ' '  + str(mean_pure_ratio2) + "\n")

    best_acc = 0.0

    # training
    for epoch in range(1, args.n_epoch):
        # train models
        cnn1.train()
        adjust_learning_rate(optimizer1, epoch)
        cnn2.train()
        adjust_learning_rate(optimizer2, epoch)
        cnn3.train()
        adjust_learning_rate(optimizer3, epoch)
        
        train_acc1, train_acc2, train_acc3, pure_ratio_1_list, pure_ratio_2_list, pure_ratio_3_list=train(train_loader, epoch, cnn1, optimizer1, cnn2, optimizer2, cnn3, optimizer3)
        # evaluate models
        test_acc1, test_acc2, test_acc3 = evaluate(test_loader, cnn1, cnn2, cnn3)

        if test_acc1 > best_acc:
            best_acc = test_acc1
            torch.save(cnn1.state_dict(), '{}_best_epoch.pth'.format(args.dataset))
        if test_acc2 > best_acc:
            best_acc = test_acc2
            torch.save(cnn2.state_dict(), '{}_best_epoch.pth'.format(args.dataset))
        if test_acc3 > best_acc:
            best_acc = test_acc3
            torch.save(cnn3.state_dict(), '{}_best_epoch.pth'.format(args.dataset))

        # save results
        mean_pure_ratio1 = sum(pure_ratio_1_list)/len(pure_ratio_1_list)
        mean_pure_ratio2 = sum(pure_ratio_2_list)/len(pure_ratio_2_list)
        mean_pure_ratio3 = sum(pure_ratio_3_list)/len(pure_ratio_3_list)
        print('Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f %% Model3 %.4f %%, Pure Ratio 1 %.4f %%, Pure Ratio 2 %.4f %% Pure Ratio 2 %.4f %%' % (epoch+1, args.n_epoch, len(test_dataset), test_acc1, test_acc2, test_acc3, mean_pure_ratio1, mean_pure_ratio2, mean_pure_ratio3))
        with open(txtfile, "a") as myfile:
            output = '{:05d}: {:10.4f} {:10.4f} {:10.4f} {:9.4f} {:9.4f} {:9.4f} {:11.4f} {:11.4f} {:11.4f}\n'.format(epoch, train_acc1, train_acc2, train_acc3, test_acc1, test_acc2, test_acc3, mean_pure_ratio1, mean_pure_ratio2, mean_pure_ratio2)
            myfile.write(output)
            # myfile.write(str(int(epoch)) + ': '  + str(train_acc1) +' '  + str(train_acc2) +' '  + str(test_acc1) + " " + str(test_acc2) + ' ' + str(mean_pure_ratio1) + ' ' + str(mean_pure_ratio2) + "\n")
    print('best acc: {}'.format(best_acc))
    with open(txtfile, "a") as myfile:
        myfile.write('===>>> best acc: {} <<<==='.format(best_acc))

if __name__=='__main__':
    main()

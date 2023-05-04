#!/usr/bin/env python3 -u
#encoding=utf-8

import argparse
import pickle
import time
import csv
import os

import numpy as np

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import random

import models
from utils import progress_bar, chunks, save_fig
import import_config

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

config = import_config.get_config()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

criterion = nn.CrossEntropyLoss()
best_acc = 0

# args\.(.*?)([): }\]])   config['$1']$2
def label_to_onehot(target, num_classes=config['nclass']):
    '''Returns one-hot embeddings of scaler labels'''
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(
        0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))

def mixup_criterion(pred, ys, num_class=config['nclass']):
    '''Returns mixup loss'''
    mixy = ys
    l = cross_entropy_for_onehot(pred, mixy)
    return l

def vec_mul_ten(vec, tensor):
    size = list(tensor.size())
    size[0] = -1
    size_rs = [1 for i in range(len(size))]
    size_rs[0] = -1
    vec = vec.reshape(size_rs).expand(size)
    res = vec * tensor
    return res

def get_lams(epoch):
    lams = pickle.load(open('lams.pickle', 'rb'))
    return lams

    result = []
    for _ in range(epoch): # epoch
        lams = np.random.normal(0, 1, size=(50000, config['klam']))
        for i in range(50000):
            lams[i] = np.abs(lams[i]) / np.sum(np.abs(lams[i]))
            if config['klam'] > 1:
                while lams[i].max() > config['upper']:     # upper bounds a single lambda
                    lams[i] = np.random.normal(0, 1, size=(1, config['klam']))
                    lams[i] = np.abs(lams[i]) / np.sum(np.abs(lams[i]))
        lams = torch.from_numpy(lams).float()
        result.append(lams)
    return result

def get_indexs(epoch):
    indexs = pickle.load(open('indexs.pickle', 'rb'))
    return indexs

    result = []
    for _ in range(epoch):
        indexs = []
        for i in range(3):
            index = torch.randperm(50000)
            indexs.append(index)
        result.append(indexs)
    return result

def get_signs(epoch):
    result = []
    for _ in range(epoch): # epoch
        sign = torch.randint(2, size=[1]) * 2.0 - 1
        result.append(sign)

    return result

    signs = pickle.load(open('signs.pickle', 'rb'))
    return signs

    result = []
    for _ in range(epoch): # epoch
        sign = torch.randint(2, size=[50000, 3, 32, 32]) * 2.0 - 1
        result.append(sign)

    return result

count = -1
def mixup_data(x, y, use_cuda, lams, indexs, signs):
    '''Returns mixed inputs, lists of targets, and lambdas'''
    # 使用分享的lams
    global count
    count += 1

    mixed_x = vec_mul_ten(lams[:, 0], x)
    mixy = vec_mul_ten(lams[:, 0], y)

    for i in range(1, config['klam']):
        index = indexs[i - 1].to(device)
        mixed_x += vec_mul_ten(lams[:, i], x[index, :])
        mixy += vec_mul_ten(lams[:, i], y[index, :])

    # if count == 1:
    #     print(x[0][0][0])
    #     print(mixed_x[0][0][0])
    #     print(mixy[0])
    #     exit()

    # if config['mode'] == 'instahide':
    #     sign = signs
    #     mixed_x *= sign.float().to(device)
    return mixed_x, lams, mixy

def generate_sample(trainloader, lams, indexs, signs):
    assert len(trainloader) == 1        # Load all training data once
    for _, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        # time1 = time.time()
        mix_inputs, lams, mixy = mixup_data(
            inputs, targets.float(), use_cuda, lams, indexs, signs)
        # time2 = time.time()
        # print('==========================')
        # print('time: {}'.format(time2 - time1))
        # exit()

        # print(lams[:, 0][0], 0)
        # for i in range(1, 4):
        #     index = indexs[i - 1]
        #     print(lams[:, i][0], index[0])

        # pickle.dump(inputs, open('test/origin.data', 'wb'))
        # pickle.dump(mix_inputs, open('test/mixup.data', 'wb'))
        # exit(0) 用来生成Instahide效果图
    return (mix_inputs, mixy)

def train(net, optimizer, inputs_all, mixy, lams, epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss, correct, total = 0, 0, 0

    seq = random.sample(range(len(inputs_all)), len(inputs_all))
    bl = list(chunks(seq, config['batch-size']))

    for batch_idx in range(len(bl)):
        b = bl[batch_idx]
        inputs = torch.stack([inputs_all[i] for i in b])
        TMP = torch.stack([mixy[i] for i in b])
        
        if config['mode'] == 'instahide' or config['mode'] == 'mixup':
            lam_batch = torch.stack([lams[i] for i in b])

        inputs = Variable(inputs)
        outputs = net(inputs)
        loss = mixup_criterion(outputs, TMP)
        
        train_loss += loss.data.item()
        total += config['batch-size']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar(batch_idx, len(inputs_all)/config['batch-size']+1,
                     'Loss: %.3f' % (train_loss / (batch_idx + 1)))
    return (train_loss / batch_idx, 100. * correct / total)

def test(net, optimizer, testloader, epoch, start_epoch):
    global best_acc
    net.eval()
    test_loss, correct_1, correct_5, total = 0, 0, 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.data.item()
            _, pred = outputs.topk(5, 1, largest=True, sorted=True)
            total += targets.size(0)
            correct = pred.eq(targets.view(targets.size(0), -
                                           1).expand_as(pred)).float().cpu()
            correct_1 += correct[:, :1].sum()
            correct_5 += correct[:, :5].sum()

            progress_bar(
                batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (test_loss /
                    (batch_idx + 1), 100. * correct_1 / total, correct_1, total))

    acc = 100. * correct_1 / total
    if epoch == start_epoch + config['epoch'] - 1 or acc > best_acc:
        save_checkpoint(net, acc, epoch)
    if acc > best_acc:
        best_acc = acc
    return (test_loss / batch_idx, 100. * correct_1 / total)

def save_checkpoint(net, acc, epoch):
    """ Save checkpoints. """
    print('Saving..')
    state = {
        'net': net,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    ckptname = os.path.join(
        './checkpoint/', f"{config['model']}_{config['data']}_{config['mode']}_{config['klam']}_{config['name']}_{config['seed']}.t7")
    torch.save(state, ckptname)

def adjust_learning_rate(optimizer, epoch):
    """ Decrease learning rate at certain epochs. """
    lr = config['lr']
    if config['data'] == 'cifar10':
        if epoch >= 100:
            lr /= 10
        if epoch >= 150:
            lr /= 10
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def get_loaders():
    cifar_normalize = transforms.Normalize(
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    if config['augment']:
        transform_cifar_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            cifar_normalize
        ])
    else:
        transform_cifar_train = transforms.Compose([
            transforms.ToTensor(),
            cifar_normalize
        ])

    transform_cifar_test = transforms.Compose([
        transforms.ToTensor(),
        cifar_normalize
    ])

    if config['data'] == 'cifar10':
        trainset = datasets.CIFAR10(root='./data',
                                    train=True,
                                    download=True,
                                    transform=transform_cifar_train)
        testset = datasets.CIFAR10(root='./data',
                                   train=False,
                                   download=True,
                                   transform=transform_cifar_test)
        num_class = 10
    
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=len(trainset),
                                              shuffle=True,
                                              num_workers=8)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=config['batch-size'],
                                             shuffle=False,
                                             num_workers=8)

    return trainloader, testloader

def log_init():
    if not os.path.isdir('results'):
        os.mkdir('results')

    logname = f"results/log_{config['model']}_{config['data']}_{config['mode']}_{config['klam']}_{config['name']}_{config['seed']}.csv"
    if not os.path.exists(logname):
        with open(logname, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter='\t')
            logwriter.writerow([
                'Epoch', 'Train loss', 'Test loss',
                'Test acc'
            ])
    return logname

def get_optimizer(net):
    return optim.SGD(net.parameters(),
                     lr=config['lr'],
                     momentum=0.9,
                     weight_decay=config['decay'])

def get_loaders_from_client():
    trainloader =  pickle.load(open('trainloader.pickle', 'rb'))
    #testloader = pickle.load(open('testloader.pickle', 'r'))
    testloader = 1
    return trainloader, testloader

def main():
    global best_acc
    total_time1 = time.time()
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    if config['seed'] != 0:
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])

    print('==> Number of lambdas: %g' % config['klam'])

    ## --------------- Prepare data --------------- ##
    print('==> Preparing data..')

    trainloader, testloader = get_loaders()
    trainloader, _ = get_loaders_from_client()

    if config['resume']:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(
            'checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/' + config['data'] + '_' +
                                config['name'] + 'ckpt.t7')
        net = checkpoint['net']
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1
        rng_state = checkpoint['rng_state']
        torch.set_rng_state(rng_state)
    else:
        print('==> Building model..')
        net = models.__dict__[config['model']](num_classes = config['nclass'])

    logname = log_init()

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
        print('==> Using CUDA..')

    optimizer = get_optimizer(net)

    epoch = config['epoch'] - start_epoch
    lams = get_lams(epoch)
    indexs = get_indexs(epoch)
    signs = get_signs(epoch)

    for epoch in range(start_epoch, config['epoch']):
        num = epoch - start_epoch

        mix_inputs_all, mixy = generate_sample(trainloader, lams[num].to(device), indexs[num], signs[num].to(device))

        train_loss, _ = train(
            net, optimizer, mix_inputs_all, mixy, lams[num], epoch)

        test_loss, test_acc1, = test(
            net, optimizer, testloader, epoch, start_epoch)


        adjust_learning_rate(optimizer, epoch)
        with open(logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter='\t')
            logwriter.writerow(
                [epoch, train_loss, test_loss, test_acc1])
                
        if epoch in [4, 9, 14]:
            total_time2 = time.time()
            print('epcoch: {}, total time: {}'.format(epoch, total_time2 - total_time1))

if __name__ == '__main__':
    main()

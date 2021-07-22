import os
import sys
import shutil
import numpy as np
import time, datetime
import torch
import random
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data.distributed

#sys.path.append("../")
from utils import *
from torchvision import datasets, transforms
import torchvision
from torch.autograd import Variable
#from birealnet import birealnet18
#from model import ResNet18_XNOR
from model import ResNet20_XNOR

import time
from torchensemble.fusion import FusionClassifier
from torchensemble.voting import VotingClassifier
from torchensemble.bagging import BaggingClassifier
from torchensemble.gradient_boosting import GradientBoostingClassifier
from torchensemble.snapshot_ensemble import SnapshotEnsembleClassifier

from torchensemble.utils.logging import set_logger

from torchsummary import summary
import warnings
warnings.filterwarnings("ignore")

ensemble = True

def display_records(records, logger):
#{{{
    msg = (
        "{:<28} | Testing Acc: {:.2f} % | Training Time: {:.2f} s |"
        " Evaluating Time: {:.2f} s"
    )

    print("\n")
    for method, training_time, evaluating_time, acc in records:
        logger.info(msg.format(method, acc, training_time, evaluating_time))
#}}}

def save_state(model, acc):
#{{{
    print('==> Saving model ...')
    state = {
            'acc': acc,
            'state_dict': model.state_dict(),
            }
    for key in state['state_dict'].keys():
        if 'module' in key:
            state['state_dict'][key.replace('module.', '')] = \
                    state['state_dict'].pop(key)
    torch.save(state, 'models/'+args.arch+'.best.pth.tar')
#}}}

if __name__=='__main__':
#{{{
    # Training settings
    parser = argparse.ArgumentParser(description='XNOR-Net-ResNet CIFAR100 Example')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
            help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=60, metavar='N',
            help='number of epochs to train (default: 60)')
    parser.add_argument('--learning_rate', type=float, default=0.01, metavar='LR',
            help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
            help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', '--wd', default=1e-5, type=float,
            metavar='W', help='weight decay (default: 1e-5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
            help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
            help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
            help='how many batches to wait before logging training status')
    parser.add_argument('--arch', action='store', default='ResNet20_XNOR',
            help='the network structure: default ResNet20_XNOR')
    parser.add_argument('--pretrained', action='store', default=None,
            help='pretrained model')
    parser.add_argument('--evaluate', action='store_true', default=False,
            help='whether to run evaluation')
    parser.add_argument('--outputfile', action='store', default='nin.out',
            help='output file')
    parser.add_argument('--data', default='data', metavar='DIR', help='path to dataset')
    parser.add_argument('--workers', type=int, default=12, metavar='N', help='num workers')
    parser.add_argument('--scheme', action='store', default='fusion',
            help='ensemble scheme')
    parser.add_argument('--n_estimators', type=int, default=2, metavar='N', help='number of estimators')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
#}}}

    print(args)
    records = []
    
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    # load data
    kwargs = {'num_workers': 40, 'pin_memory': True} if args.cuda else {}

    print('==> Preparing data..')
    transform_train = transforms.Compose([
         transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
     ])
 
    transform_test = transforms.Compose([
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
     ])
 
    trainset = torchvision.datasets.CIFAR100(root=args.data, train=True, 
      download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, 
      batch_size=args.batch_size, shuffle=True, 
      num_workers=args.workers, pin_memory=True)
 
    testset = torchvision.datasets.CIFAR100(root=args.data, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, 
      batch_size=args.batch_size,
      shuffle=False, num_workers=args.workers, pin_memory=True)

    logger = set_logger("classification_xnor-net-resnet_cifar100")

    # generate the model
    if args.arch == 'ResNet20_XNOR':
      #model = LeNet_5.LeNet_5()
      net =  ResNet20_XNOR()
      print(net)
      if ensemble == True:
       #HJKIM: This is the key!!!
       #freeze weights in convolution layer in XNOR-Net-ResNet20
        net.conv1.weight.requires_grad = False
        net.layer1[0].binaryconv1.weights.requires_grad = False
        net.layer1[0].binaryconv2.weights.requires_grad = False
        net.layer1[1].binaryconv1.weights.requires_grad = False
        net.layer1[1].binaryconv2.weights.requires_grad = False
        net.layer1[2].binaryconv1.weights.requires_grad = False
        net.layer1[2].binaryconv2.weights.requires_grad = False
        net.layer2[0].binaryconv1.weights.requires_grad = False
        net.layer2[0].binaryconv2.weights.requires_grad = False
        net.layer2[0].shortcut[0].weight.requires_grad = False #conv2d shortcut
        net.layer2[1].binaryconv1.weights.requires_grad = False
        net.layer2[1].binaryconv2.weights.requires_grad = False
        net.layer2[2].binaryconv1.weights.requires_grad = False
        net.layer2[2].binaryconv2.weights.requires_grad = False
        net.layer3[0].binaryconv1.weights.requires_grad = False
        net.layer3[0].binaryconv2.weights.requires_grad = False
        net.layer3[0].shortcut[0].weight.requires_grad = False #conv2d shortcut
        net.layer3[1].binaryconv1.weights.requires_grad = False
        net.layer3[1].binaryconv2.weights.requires_grad = False
        net.layer3[2].binaryconv1.weights.requires_grad = False
        net.layer3[2].binaryconv2.weights.requires_grad = False
        net.linear.weight.requires_grad = False
    else:
      print('ERROR: specified arch is not suppported')
      exit()

    if not args.pretrained:
       best_acc = 0.0
    else:
      checkpoint = torch.load(args.pretrained)['state_dict']
      for key in list(checkpoint.keys()):
        if 'module.' in key:
          checkpoint[key.replace('module.', '')] = checkpoint[key]
          del checkpoint[key]
      net.load_state_dict(checkpoint)

    if args.cuda:
      net.cuda()
      net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

    if ensemble == True:  
      print("=====================================================")
      print("======== This is Emsemble Mode! =========")
      if args.scheme == 'fusion':
        model = FusionClassifier(
          estimator=net, n_estimators=args.n_estimators, cuda=True)
      elif args.scheme == 'voting':
        model = VotingClassifier(
          estimator=net, n_estimators=args.n_estimators, cuda=True)
      elif args.scheme == 'bagging':
        model = BaggingClassifier(
          estimator=net, n_estimators=args.n_estimators, cuda=True)
      elif args.scheme == 'gradientboosting':
        model = GradientBoostingClassifier(
          estimator=net, n_estimators=args.n_estimators, cuda=True)
      elif args.scheme == 'snapshotensemble':
        model = SnapshotEnsembleClassifier(
          estimator=net, n_estimators=args.n_estimators, cuda=True)

      
    #HJKIM
    model.set_optimizer("Adam", lr=args.learning_rate, weight_decay=args.weight_decay)

    model.set_scheduler("LambdaLR", lr_lambda=lambda step : (1.0 -step/args.epochs))

    criterion = nn.CrossEntropyLoss()

    #Training
    tic = time.time()
    model.fit(train_loader, epochs=args.epochs, test_loader=test_loader, save_dir="models")
    toc = time.time()
    training_time = toc - tic

    # Evaluating
    tic = time.time()
    testing_acc = model.evaluate(test_loader)
    toc = time.time()
    evaluating_time = toc - tic

    if args.scheme == 'fusion':
      records.append(
        ("FusionClassifier", training_time, evaluating_time, testing_acc))
    elif args.scheme == 'voting':
      records.append(
        ("VotingClassifier", training_time, evaluating_time, testing_acc))
    elif args.scheme == 'bagging':
      records.append(
        ("BaggingClassifier", training_time, evaluating_time, testing_acc))
    elif args.scheme == 'gradientboosting':
      records.append(
        ("GradientBoostingClassifier", training_time, evaluating_time, testing_acc))
    elif args.scheme == 'snapshotensemble':
      records.append(
        ("SnapshotEnsembleClassifier", training_time, evaluating_time, testing_acc))

    display_records(records, logger)




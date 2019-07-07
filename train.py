import torch
from torchvision import datasets, transforms, models
from collections import OrderedDict
from torch import nn
import torch.optim as optim
import numpy as np

import argparse

import functions

parser = argparse.ArgumentParser(description='Train.py')

parser.add_argument('data_dir', action='store', help='Path to training data')
parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
parser.add_argument('--gpu', dest="gpu", action="store", default='cuda:0')
parser.add_argument('--learning_rate', dest="lr", action="store", default=0.0003, type=int)
parser.add_argument('--dropout', dest="dropout", action="store", default=0.05, type=int)
parser.add_argument('--epochs', dest="epochs", action="store", default=3, type=int)
parser.add_argument('--arch', dest="arch", action="store", default="vgg16")
parser.add_argument('--hidden_units', dest="hidden_units", action="store", default=512, type=int)

results = parser.parse_args()

data_dir = results.data_dir
save_dir = results.save_dir
gpu= results.gpu
lr = results.lr
dropout = results.dropout
epochs = results.epochs
network = results.arch
hidden_units = results.hidden_units

trainloader, validloader, testloader, train_data = functions.load_data(data_dir)
model, criterion, optimizer = functions.network(network, dropout, hidden_units, lr, gpu)

functions.train_network(model, criterion, optimizer, trainloader, gpu, epochs, validloader)

functions.save_model(save_dir, model, epochs, optimizer, train_data, network, hidden_units, dropout)

print('Compleate')
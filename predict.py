import torch
from torchvision import datasets, transforms, models
from collections import OrderedDict
from torch import nn
import torch.optim as optim
import numpy as np

import argparse

import functions

import json


parser = argparse.ArgumentParser(description='Predict.py')

parser.add_argument('img_file', action='store')
parser.add_argument('checkpoint', action="store", type=str)
parser.add_argument('--gpu', dest="gpu", action="store", default='cuda:0')
parser.add_argument('--top_k', dest="top_k", action="store", default=5, type=int)
parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')

results = parser.parse_args()

image_path = results.img_file
checkpoint = results.checkpoint
gpu = results.gpu
topk = results.top_k
cat = results.category_names


model = functions.load_checkpoint(checkpoint)

with open(cat, 'r') as f:
    cat_to_name = json.load(f)
    
probs, classes = functions.predict(image_path, model, topk)

print(probs)
print(classes) 

names = []
for i in classes:
    names += [cat_to_name[i]]


print("Name: {} Probability {}% ".format(names[0], round(100 * acc / total,3)))

import time
import os
import sys
import csv
import json
import shutil
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import clear_output

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader as dataloader
import torchvision.models as models

sys.path.insert(1, 'HelperScripts')
from HelperFunctions import *
from pytorch_resnet import *



#the size of our mini batches
batch_size = 64
#Size of downsampled image
image_size = 64

device_name = torch.cuda.get_device_name(0)
cuda_device = torch.device("cuda:0")
cpu_device = torch.device("cpu:0")

classes = ('Basophil','Monocyte','Lymphocyte','Eosinophil','Neutrophil')

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
            transforms.CenterCrop(250),
            #SquarePad(max_wh=),
            #transforms.Resize((image_size,image_size), max_size=None, antialias=None),
            transforms.ToTensor()
            #transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

train_data = datasets.ImageFolder("Data/WBC_Classification/fluor", transform=transform)
validation_split = 0.9
n_train_examples = int(len(train_data)*validation_split)
n_valid_examples = len(train_data) - n_train_examples
train_data, valid_data = torch.utils.data.random_split(train_data, [n_train_examples, n_valid_examples])
print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
train_loader = dataloader(train_data, shuffle=True, batch_size=batch_size, pin_memory=True, drop_last=True)
valid_loader = dataloader(valid_data, batch_size=batch_size, pin_memory=True, drop_last=True)
print(f'Length of trainloader: {len(train_loader)}')

for i in range(valid_data.__len__()):
    x,y = valid_data.__getitem__(i)
    img = np.transpose(x.cpu().numpy(), (1,2,0))
    img = (img*255).astype(np.uint8)
    Image.fromarray(img, mode='RGB').save("Data/WBC_Classification_split/fluor/test/{}/img{}.png".format(classes[y],i))
for i in range(train_data.__len__()):
    x,y = train_data.__getitem__(i)
    img = np.transpose(x.cpu().numpy(), (1,2,0))
    img = (img*255).astype(np.uint8)
    Image.fromarray(img, mode='RGB').save("Data/WBC_Classification_split/fluor/train/{}/img{}.png".format(classes[y],i))
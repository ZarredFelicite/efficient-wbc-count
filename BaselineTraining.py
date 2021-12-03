import time
import os
import sys
import csv
import math
import json
import shutil
import random
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import clear_output

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.profiler
import torchvision.datasets as datasets
from torch.utils.data import DataLoader as dataloader
import torchvision.models as models

sys.path.insert(1, 'utils')
from HelperFunctions import *
from pytorch_resnet import *
from randaugment import RandAugmentMC

from efficientnet_lite import build_efficientnet_lite

#Logging
os.environ["WANDB_SILENT"] = "true"
import wandb
wandb.login()

if __name__ == '__main__':

    cuda_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu:0")


    #This function should perform a single training epoch using our training data
    def train(model, device, train_loader, valid_loader, optimizer, loss_fun, epoch, valid_loss, valid_acc, early_stopping, scheduler, prof):
        epoch_loss = 0
        epoch_acc = 0
        last_loss = 0
        last_acc = 0
        model.train()
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            fx = model(x)
            loss = loss_fun(fx, y)
            acc = calculate_accuracy(fx, y)
            loss.backward()
            optimizer.step()
            if prof is not None:
                prof.step()
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            wait = len(train_loader)-1
            if i % wait == wait-1:
                last_loss = epoch_loss / wait
                last_acc = epoch_acc / wait
                #print(' batch {} loss: {:.4f} acc: {:.4f}'.format(i+1, last_loss, last_acc))
                tb_x = epoch*len(train_loader) + i + 1
                valid_loss, valid_acc, early = eval(model, device, valid_loader, loss_fun, early_stopping=early_stopping)
                wandb.log({"Training vs. Validation Loss": {"Training" : last_loss, "Validation" : valid_loss}}, step=tb_x)
                wandb.log({"Training vs. Validation Accuracy": {"Training" : last_acc, "Validation": valid_acc}}, step=tb_x)
                epoch_loss = 0.
                epoch_acc = 0.
                if early:
                    break
            scheduler.step()
        #clear_output(True)
        return last_loss, early, valid_acc, valid_loss

    def training(model, cuda_device, config, prof):
        
        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=config['patience'], verbose=True, delta=config['delta'], path=config['save_path'])
        loss_fn = nn.CrossEntropyLoss().to(cuda_device)
        learning_rate = config['lr']
        original_lr = learning_rate
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = learning_rate, betas=(0.9, 0.999), weight_decay=1e-10)
        #Learning rate scheduling
        #lr_lambda = lambda epoch: 0.998 #** (epoch*0.5)
        #lr_lambda = config['lr_lambda']
        #scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=len(train_loader), epochs=config['epochs'], div_factor=4, pct_start=0.25, three_phase=True, final_div_factor=1e8)
        start_epoch = 0
        num_epochs = config['epochs']
        for epoch in range(start_epoch, num_epochs):
            if epoch==0:
                valid_loss=0
                valid_acc=0
            loss, early, valid_acc, valid_loss = train(model, cuda_device, train_loader, valid_loader, optimizer, loss_fn, epoch, valid_loss, valid_acc, early_stopping, scheduler, prof)
            print("Epoch: {}  Acc: {}  Loss: {}  lr: {}".format(epoch,valid_acc,valid_loss,scheduler.get_last_lr()[0]/original_lr))
            wandb.log({"Learning Rate": scheduler.get_last_lr()[0]}, step=(epoch+1)*len(train_loader))
            #scheduler.step()
            if early:
                break
        #clear_output(True)

        check_point = torch.load(config['save_path'])
        model.load_state_dict(check_point['model_state_dict'])
    #'Green','Purple','Red'
    for colour in ['Green','Purple','Red','Fluor']:
        config_train = dict(
            epochs=20,
            classes=['Basophil','Eosinophil','Lymphocyte','Monocyte','Neutrophil'],
            image_size=160,
            batch_size=64,
            valid_batch_size = 12,
            batch_size_latency=1,
            save = 'Models/baseline_nopretrain/',
            save_suffix = '_baseline_np_'+colour,
            lr = 1.5e-4,
            patience=15,
            delta=0.001,
            lr_lambda = 0.9,
            transforms = dict(n=15,m=10,norm=None),
            architecture_list = ['resnet18',
                             # 'resnet34',
                             # 'resnet50', 
                             #'mobilenet_v3_small'
                             #'mobilenet_v2',
                             # 'shufflenet_v2_x0_5',
                             # 'regnet_x_400mf',
                             # 'regnet_x_800mf',
                             # 'regnet_y_400mf',
                             # 'regnet_y_800mf',
                             #'efficientnet_lite'
                             ],
            data_path = 'T:/Haematology FYP/Data/WBC_Classification_3172_c/',
            tags = [colour,'Training','raabin_pre'],
            profiling = False,
            logging = 'online',#'disabled'
            num_workers=4
        )
        #['efficientnet_b0','efficientnet_b1','efficientnet_b2','regnet_y_400mf','regnet_y_800mf'],
        # ['resnet18','resnet34','resnet50', 'mobilenet_v2', 'mobilenet_v3_small', 'mnasnet1_0', 'shufflenet_v2_x0_5', 'efficientnet_b0','efficientnet_b1','efficientnet_b2','regnet_x_400mf','regnet_x_800mf','regnet_y_400mf','regnet_y_800mf']
        #best
        #['resnet18','resnet34','resnet50', 'mobilenet_v2', 'mobilenet_v3_small', 'shufflenet_v2_x0_5','regnet_x_400mf','regnet_x_800mf','regnet_y_400mf','regnet_y_800mf']


        ##########################################################
        #DATA PREPARATION
        ##########################################################
        
        transform = transforms.Compose([
                    RandAugmentMC(n=config_train['transforms']['n'], m=config_train['transforms']['m']),
                    transforms.ToTensor()
        ])
        transform_test = transforms.Compose([
                    transforms.ToTensor()
        ])


        train_data = datasets.ImageFolder(config_train['data_path']+colour+'/train', transform=transform)
        test_data = datasets.ImageFolder(config_train['data_path']+colour+'/val', transform=transform_test)
        validation_split = 0.9
        n_train_examples = int(len(train_data))
        n_valid_examples = len(test_data)
        train_loader = dataloader(train_data, shuffle=True, batch_size=config_train['batch_size'], num_workers=config_train['num_workers'], pin_memory=True, drop_last=True, persistent_workers=True)
        valid_loader = dataloader(test_data, batch_size=config_train['valid_batch_size'], num_workers=config_train['num_workers'], pin_memory=True, drop_last=True, persistent_workers=True)
        config_train['train_size'] = n_train_examples
        config_train['val_size'] = n_valid_examples
        print("Train samples: {}    Batch size: {}".format(n_train_examples, len(train_loader)))
        print("Validation samples: {}     Batch size: {}".format(n_valid_examples,len(valid_loader)))
        torch.backends.cudnn.benchmark = True
        ######################################################
        #TRAINING
        ######################################################
        for architecture in config_train['architecture_list']:
            with wandb.init(project="Baseline Training", config=config_train, group='Training', tags=config_train['tags'], job_type='train', reinit=True, mode=config_train['logging']):
                #%%wandb
                wandb.config['architecture'] = architecture
                wandb.run.summary['architecture'] = architecture
                wandb.config['colour'] = colour
                wandb.run.name = wandb.config['architecture']+wandb.config['save_suffix']
                wandb.config['file'] = wandb.run.name + ".pt"
                wandb.config['save_path'] = wandb.config['save'] + wandb.config['file']
                config = wandb.config
                if config['profiling']:
                    prof = torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA],
                        schedule=torch.profiler.schedule(wait=1, warmup=1, active=2),
                        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/'+ architecture), profile_memory=True)
                    prof.start()
                else:
                    prof = None
                if architecture == 'efficientnet_lite':
                    model = build_efficientnet_lite('efficientnet_lite0', 5).to(cuda_device)
                else:
                    model = create_model(architecture, cuda_device, pretrained=False, freeze=False)
                #Custom pretraining weights
                check_point = torch.load('Models/raabin/Resnet_18_v1.2.pt')
                model.load_state_dict(check_point['model_state_dict'])
                #Training
                training(model, cuda_device, config, prof)

                if prof is not None:
                    prof.stop()
                #Export model to ONNX
                # images, _ = next(iter(train_loader))
                # images = images.to(cuda_device)
                # onnx_file = config['architecture'] + config['save_suffix'] + '.onnx'
                # torch.onnx.export(model, images, onnx_file)
                # wandb.save(onnx_file)
                torch.save({'model_state_dict' : model.state_dict()}, os.path.join(wandb.run.dir, wandb.config['file']))
                model.eval()
                #Model size
                wandb.run.summary['Model Size'] = size_measure(model)
                #Measure Accuracy
                loss_fn = nn.CrossEntropyLoss().to(cuda_device)
                results, cm = evaluate2(model=model, device=cuda_device, loader=valid_loader, batch_size=config['valid_batch_size'], loss_fun=loss_fn, labels_indx=list(test_data.class_to_idx.values()), names=list(test_data.class_to_idx.keys()))
                # sample_weight=torch.unique(torch.FloatTensor(test_data.targets),return_counts=True)[1].tolist()
                #pprint.pprint(results)
                wandb.run.summary['CM'] = cm
                wandb.log({'CM Plot' : wandb.Image(return_cm(cm, config['classes'], normalize=True, title='Confusion matrix '+ config['architecture'] + config['save_suffix'], cmap=plt.cm.Blues))})
                wandb.run.summary['Results'] = results
                #Measure Latency
                time_logger, std_logger, time2_logger = latency_test(model, cpu_device, repetitions=300, batch_size=config['batch_size_latency'], num_threads=0)
                wandb.run.summary['Inference Time'] = {'CUDA':time_logger,'Std':std_logger,'CPU':time2_logger}
                wandb.run.summary['Acc/Time'] = results['accuracy']/time_logger
                wandb.run.summary['Macro F1/Time'] = results['macro avg']['f1-score']/time_logger
                torch.cuda.empty_cache()
                clear_output(wait=True)
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
import torch.quantization
from torch.quantization import QuantStub, DeQuantStub

sys.path.insert(1, 'HelperScripts')
from HelperFunctions import *
from pytorch_resnet import *
from randaugment import RandAugmentMC

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
            scheduler.step()
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
        #clear_output(True)
        return last_loss, early, valid_acc, valid_loss

    def training(model, cuda_device, config, prof):
        save_path = config['save'] + config['architecture'] + config['save_suffix'] + ".pt"
        wandb.config['file'] = save_path
        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=config['patience'], verbose=True, delta=config['delta'], path=save_path)
        loss_fn = nn.CrossEntropyLoss().to(cuda_device)
        learning_rate = config['lr']
        original_lr = learning_rate
        optimizer = optim.Adam(model.parameters(), lr = learning_rate, betas=(0.9, 0.999), weight_decay=1e-10)
        #Learning rate scheduling
        #lr_lambda = lambda epoch: 0.9** (epoch*0.5)
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

        check_point = torch.load(save_path)
        model.load_state_dict(check_point['model_state_dict'])
    #'Green','Purple','Red'
    for colour in ['Green','Purple','Red','Fluor']:
        config_train = dict(
            epochs=30,
            classes=['Basophil','Eosinophil','Lymphocyte','Monocyte','Neutrophil'],
            quant_type = 'PTQ',
            observer = ['MovingAverageMinMaxObserver','MinMaxObserver'],
            dtype = ["quint8", "quint8"],
            qscheme = ['per_tensor_affine', 'per_tensor_symmetric'],
            reduce_range = ["False", "False"],
            image_size=160,
            batch_size=64,
            valid_batch_size = 12,
            batch_size_latency=1,
            save = 'Models/qat/',
            save_suffix = '_qat_'+colour,
            jit_name = '_qat_jit_'+ colour + '.pt',
            lr = 1e-5,
            patience=15,
            delta=0.001,
            lr_lambda = 0.9,
            transforms = dict(n=15,m=10,norm=None),
            architecture_list = [#'resnet18',
                              #'resnet34',
                              #'resnet50', 
                             # 'regnet_y_400mf',
                             #'mobilenet_v3_small'
                             'mobilenet_v2',
                             # 'shufflenet_v2_x0_5',
                             # 'regnet_x_400mf',
                             # 'regnet_x_800mf',
                             # 'regnet_y_400mf',
                             # 'regnet_y_800mf'
                             ],
            data_path = 'T:/Haematology FYP/Data/WBC_Classification_3172_c/',
            tags = [colour,'Training','v3'],
            profiling = False,
            num_workers=6
        )
        #['efficientnet_b0','efficientnet_b1','efficientnet_b2','regnet_y_400mf','regnet_y_800mf'],
        # ['resnet18','resnet34','resnet50', 'mobilenet_v2', 'mobilenet_v3_small', 'mnasnet1_0', 'shufflenet_v2_x0_5', 'efficientnet_b0','efficientnet_b1','efficientnet_b2','regnet_x_400mf','regnet_x_800mf','regnet_y_400mf','regnet_y_800mf']
        #best
        #['resnet18','resnet34','resnet50', 'mobilenet_v2', 'mobilenet_v3_small', 'shufflenet_v2_x0_5','regnet_x_400mf','regnet_x_800mf','regnet_y_400mf','regnet_y_800mf']


        train_loader, valid_loader, test_loader, test_data = datamaker(colour,config=config_train)

        torch.backends.cudnn.benchmark = True
        ######################################################
        #TRAINING
        ######################################################
        for architecture in config_train['architecture_list']:
            with wandb.init(project="Baseline Training", config=config_train, group='Training', tags=config_train['tags'], job_type='train', reinit=True):
                #%%wandb
                wandb.config['architecture'] = architecture
                wandb.run.summary['architecture'] = architecture
                wandb.config['colour'] = colour
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
                wandb.run.name = config['architecture']+config['save_suffix']
                
                if architecture in ['resnet18','resnet34','resnet50']:
                    model = load_model(architecture).to(cpu_device).eval()
                elif architecture in ['regnet_y_400mf']:
                    model = create_model(architecture, cpu_device)
                    #model = CustomRegNet()
                elif architecture in ['mobilenet_v3_small']:
                    model = create_model(architecture, cpu_device)
                elif architecture == 'mobilenet_v2':
                    model = MobileNetV2(num_classes=5)

                #Load float model weights for pretraining
                float_model_file = 'Models/baseline/temp/' + colour + architecture + '_profiling_' + colour + '.pt'
                print('Loading weights from: {}'.format(float_model_file))
                check_point = torch.load(float_model_file)
                model.load_state_dict(check_point['model_state_dict'])
                model.to(cpu_device)

                
                #Fuse model (conv+batchnorm+relu)
                print('Fusing {}...'.format(architecture))
                if architecture in ['resnet18','resnet34','resnet50']:
                    model = model_fusion(model, architecture)
                elif architecture in ['regnet_y_400mf']:
                    fusion_regnet(model)
                elif architecture in ['mobilenet_v3_small']:
                    #model = create_model(architecture, cpu_device)
                    pass
                elif architecture == 'mobilenet_v2':
                    model.fuse_model()

                #Quantization configuration
                quantization_config = torch.quantization.QConfig(
                    activation=getattr(torch.quantization, config['observer'][0]).with_args(dtype=torch.quint8, qscheme=getattr(torch, config['qscheme'][0])),
                    weight=getattr(torch.quantization, config['observer'][1]).with_args(dtype=torch.qint8, qscheme=getattr(torch, config['qscheme'][1])))
                model.qconfig = quantization_config

                # https://pytorch.org/docs/stable/_modules/torch/quantization/quantize.html#prepare_qat
                torch.quantization.prepare_qat(model, inplace=True)

                #Quantization aware training.
                print("Calibrating QAT Model...")
                model.train()
                model.to(cuda_device)
                loss_fn = nn.CrossEntropyLoss()
                training(model, cuda_device, config, prof)
                model.to(cpu_device)

                #Converting to quantized model
                print('Converting to quantized model')
                model = torch.quantization.convert(model, inplace=True)

                #Save Model
                #State-dict (must go through quantization process without calibration to load in state-dict)
                print('Saving quantized {}...'.format(architecture))
                torch.save(model.state_dict(), config['save']+architecture+config['save_suffix'])
                #Torchscript
                scripted_model = torch.jit.script(model)
                torch.jit.save(scripted_model, config['save'] + architecture + config['jit_name'])

                if prof is not None:
                    prof.stop()
                
                model.eval()
                #Model size
                wandb.run.summary['Model Size'] = size_measure(model)
                #Measure Accuracy
                loss_fn = nn.CrossEntropyLoss().to(cpu_device)
                results, cm = evaluate2(model=model, device=cpu_device, loader=test_loader, batch_size=config['valid_batch_size'], loss_fun=loss_fn, labels_indx=list(test_data.class_to_idx.values()), names=list(test_data.class_to_idx.keys()))
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
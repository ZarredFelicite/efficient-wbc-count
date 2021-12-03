import time
import os
import sys
import csv
import json
import shutil
import random
import pprint
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
from randaugment import RandAugmentMC

os.environ["WANDB_SILENT"] = "true"
import wandb
wandb.login()
#%env "WANDB_NOTEBOOK_NAME" "Baseline Training"

if __name__ == '__main__':

    device_name = torch.cuda.get_device_name(0)
    cuda_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu:0")

    transform_test = transforms.Compose([transforms.ToTensor()])

    for colour in ['Green','Purple','Red','Fluor']:
        print(colour)
    #'green','red','purple','fluor'
        config_ptq = dict(
            classes=['Basophil','Eosinophil','Lymphocyte','Monocyte','Neutrophil'],
            image_size=160,
            batch_size=32,
            batch_size_latency=1,
            save = 'Models/baseline/temp/'+ colour,
            save_suffix = "_baseline_wbc_"+colour,
            load_suffix = '_profiling_'+colour,
            #test_set = colour + '_test',
            architecture_list = ['resnet18',
                                 'resnet34',
                                 'resnet50', 
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
            tags = [colour,'PTQ','v3'],
            logging = 'online'#'disabled' #
        )
        test_data = datasets.ImageFolder(config_ptq['data_path'] + colour + '/test', transform=transform_test)
        print(f'Number of testing examples: {len(test_data)}')
        test_loader = dataloader(test_data, batch_size=config_ptq['batch_size'], pin_memory=True, drop_last=True)
        print(test_data.class_to_idx)
        torch.unique(torch.FloatTensor(test_data.targets), return_counts=True)
        print(list(test_data.class_to_idx.values()))

        for architecture in config_ptq['architecture_list']:
            with wandb.init(project="Baseline Training", config=config_ptq, group="PTQ", tags=config_ptq['tags'], job_type='eval', reinit=True, mode=config_ptq['logging']):
                wandb.config['architecture'] = architecture
                wandb.run.summary['architecture'] = architecture
                wandb.config['colour'] = colour
                config = wandb.config
                wandb.run.name = config['architecture']+config['save_suffix']
                float_model_file = config['save']+architecture+config['load_suffix']+'.pt'
                model = create_model(architecture, cpu_device)
                #num_ftrs = model.fc.in_features
                #model.fc = nn.Linear(num_ftrs, 5).to(cpu_device)
                check_point = torch.load(float_model_file)
                model.load_state_dict(check_point['model_state_dict'])
                model.to(cpu_device)
                #Evaluation
                model.eval()
                #Model size
                wandb.run.summary['Model Size'] = size_measure(model)
                #Measure Accuracy
                loss_fn = nn.CrossEntropyLoss().to(cpu_device)
                results, cm = evaluate2(model=model, device=cpu_device, loader=test_loader, batch_size=config['batch_size'], loss_fun=loss_fn, labels_indx=list(test_data.class_to_idx.values()), names=list(test_data.class_to_idx.keys()))
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
                #print_measurements(config['classes'],precision,sensitivity,f1,acc,cm,wandb.run.summary['Model Size'],time_logger,time2_logger)
                #clear_output(wait=True)
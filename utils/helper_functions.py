import datetime
import itertools
import json
import os
import random
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.profiler as profiler
import torch.quantization
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms.functional as tF
import wandb
from matplotlib.gridspec import GridSpec
from PIL import Image
from seaborn import heatmap
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from torch.quantization import DeQuantStub, QuantStub
from torch.utils.data import DataLoader as dataloader
from tqdm import tqdm

# from randaugment import RandAugmentMC

# import warnings
# warnings.filterwarnings(
#     action='ignore',
#     category=DeprecationWarning,
#     module=r'.*'
# )
# warnings.filterwarnings(
#     action='default',
#     module=r'torch.quantization'
# )

# Global Variables
classes = ("Basophil", "Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil")


def deleter(path):
    for dir in os.listdir(path):
        newdir = path + "/" + dir
        if os.path.isdir(newdir):
            deleter(newdir)
        else:
            os.remove(newdir)


# def calculate_accuracy(fx, y):
#     preds = fx.max(1, keepdim=True)[1]
#     correct = preds.eq(y.view_as(preds)).sum()
#     acc = correct.float() / preds.shape[0]
#     return acc


# def plot_confusion_matrix(
#     cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
# ):
#     if normalize:
#         cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print("Confusion matrix, without normalization")

#     print(cm)
#     plt.imshow(cm, interpolation="nearest", cmap=cmap)
#     plt.title(title)
#     # plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)

#     fmt = ".2f" if normalize else "d"
#     thresh = cm.max() / 2.0
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(
#             j,
#             i,
#             format(cm[i, j], fmt),
#             horizontalalignment="center",
#             color="white" if cm[i, j] > thresh else "black",
#         )

#     plt.tight_layout()
#     plt.ylabel("True label")
#     plt.xlabel("Predicted label")


# def return_cm(
#     cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
# ):
#     plt.figure(figsize=(5, 5))
#     if normalize:
#         cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
#     plt.imshow(cm, interpolation="nearest", cmap=cmap)
#     plt.title(title)
#     # plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#     fmt = ".2f" if normalize else "d"
#     thresh = cm.max() / 2.0
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(
#             j,
#             i,
#             format(cm[i, j], fmt),
#             horizontalalignment="center",
#             color="white" if cm[i, j] > thresh else "black",
#         )
#     plt.tight_layout()
#     plt.ylabel("True label")
#     plt.xlabel("Predicted label")
#     save = "plots/" + title + ".png"
#     plt.savefig(save, bbox_inches="tight")
#     return save


def create_model(architecture, device='cpu', pretrained=True, quantized=False, freeze=False):
    if quantized:
        model = getattr(models.quantization, architecture)(pretrained=pretrained).to(device)
    else:
        model = getattr(models, architecture)(pretrained=pretrained).to(device)
    if architecture in [
        "resnet18",
        "resnet34",
        "resnet50",
        "resnext50",
        "shufflenet_v2_x0_5",
        "shufflenet_v2_x1_0",
        "regnet_x_400mf",
        "regnet_x_800mf",
        "regnet_y_400mf",
        "regnet_y_800mf",
    ]:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 5).to(device)
    elif architecture in [
        "mnasnet1_0",
        "mobilenet_v2",
        "mobilenet_v3_small",
        "mobilenet_v3_large",
        "vgg16",
        "efficientnet_b0",
        "efficientnet_b1",
        "efficientnet_b2",
    ]:
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, 5).to(device)
    elif architecture == "densenet121":
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 5).to(device)
    elif architecture == "squeezenet1_1":
        model.classifier[1] = nn.Conv2d(512, 5, kernel_size=(1, 1), stride=(1, 1)).to(
            device
        )
    if freeze:
        # Setting all non-classifier layers to frozen
        for n, m in model.named_children():
            if n not in ["fc", "linear", "classifier"]:
                for param in m.parameters():
                    param.requires_grad = False
    return model


def latency_test(model, device, repetitions, batch_size=1, num_threads=0):
    # Measure Latency
    model.to(device)
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    elapsed = 0
    timings = np.zeros((repetitions, 1))
    dummy_input = torch.randn(batch_size, 3, 64, 64, dtype=torch.float).to(device)
    if num_threads != 0:
        torch.set_num_threads(num_threads)
    # GPU-WARM-UP
    for _ in range(10):
        _ = model(dummy_input)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            s = datetime.datetime.now()
            starter.record()
            _ = model(dummy_input)
            ender.record()
            elapsed += (datetime.datetime.now() - s).total_seconds() * 1000
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    time_cpu = elapsed / repetitions
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    return mean_syn, std_syn, time_cpu


def size_measure(model):
    torch.save(model.state_dict(), "temp.p")
    model_size = os.path.getsize("temp.p") / 1e6
    os.remove("temp.p")
    return model_size


# def print_measurements(
#     classes, precision, sensitivity, f1, acc, cm, model_size, mean_syn, time_cpu
# ):
#     print("Performance Metrics:")
#     print("Overall Accuracy: %.2f" % (acc[len(classes)]))
#     for i in range(len(classes)):
#         print(
#             "%s: P %6.2f  S %6.2f  F1 %6.2f  A %6.2f"
#             % (classes[i].ljust(11), precision[i], sensitivity[i], f1[i], acc[i])
#         )
#     print(cm)
#     print("model size")
#     print(model_size)
#     print("Sync Latency")
#     print(mean_syn)
#     print("CPU Latency")
#     print(time_cpu)


# class EarlyStopping:
#     """Early stops the training if validation loss doesn't improve after a given patience."""

#     def __init__(
#         self, patience=7, verbose=False, delta=0, path="checkpoint.pt", trace_func=print
#     ):
#         """
#         Args:
#             patience (int): How long to wait after last time validation loss improved.
#                             Default: 7
#             verbose (bool): If True, prints a message for each validation loss improvement.
#                             Default: False
#             delta (float): Minimum change in the monitored quantity to qualify as an improvement.
#                             Default: 0
#             path (str): Path for the checkpoint to be saved to.
#                             Default: 'checkpoint.pt'
#             trace_func (function): trace print function.
#                             Default: print
#         """
#         self.patience = patience
#         self.verbose = verbose
#         self.counter = 0
#         self.best_score = None
#         self.early_stop = False
#         self.val_loss_min = np.Inf
#         self.delta = delta
#         self.path = path
#         self.trace_func = trace_func

#     def __call__(self, val_loss, model):

#         score = -val_loss

#         if self.best_score is None:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model)
#         elif score < self.best_score + self.delta:
#             self.counter += 1
#             self.trace_func(
#                 f"EarlyStopping counter: {self.counter} out of {self.patience}"
#             )
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model)
#             self.counter = 0

#     def save_checkpoint(self, val_loss, model):
#         """Saves model when validation loss decrease."""
#         if self.verbose:
#             self.trace_func(
#                 f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
#             )
#         torch.save(
#             {
#                 "model_state_dict": model.state_dict(),
#                 #'optimizer_state_dict':  optimizer.state_dict(),
#                 #'train_acc':             train_acc,
#                 #'valid_acc':             valid_acc,
#                 #'test_acc':              acc[len(classes)],
#             },
#             self.path,
#         )
#         self.val_loss_min = val_loss


# class SquarePad:
#     def __call__(self, image, max_wh):
#         # max_wh = max(image.size)
#         p_left, p_top = [(max_wh - s) // 2 for s in image.size]
#         p_right, p_bottom = [
#             max_wh - (s + pad) for s, pad in zip(image.size, [p_left, p_top])
#         ]
#         padding = (p_left, p_top, p_right, p_bottom)
#         return tF.pad(image, padding, 0, "constant")


# def model_fusion(model, architecture):
#     # Fuse the model in place rather manually.
#     if architecture == "resnet50":
#         model = torch.quantization.fuse_modules(
#             model, [["conv1", "bn1", "relu"]], inplace=True
#         )
#         for module_name, module in model.named_children():
#             if "layer" in module_name:
#                 for basic_block_name, basic_block in module.named_children():
#                     torch.quantization.fuse_modules(
#                         basic_block,
#                         [
#                             ["conv1", "bn1", "relu1"],
#                             ["conv2", "bn2", "relu2"],
#                             ["conv3", "bn3"],
#                         ],
#                         inplace=True,
#                     )
#                     for sub_block_name, sub_block in basic_block.named_children():
#                         if sub_block_name == "downsample":
#                             torch.quantization.fuse_modules(
#                                 sub_block, [["0", "1"]], inplace=True
#                             )
#     else:
#         model = torch.quantization.fuse_modules(
#             model, [["conv1", "bn1", "relu"]], inplace=True
#         )
#         for module_name, module in model.named_children():
#             if "layer" in module_name:
#                 for basic_block_name, basic_block in module.named_children():
#                     torch.quantization.fuse_modules(
#                         basic_block,
#                         [["conv1", "bn1", "relu1"], ["conv2", "bn2"]],
#                         inplace=True,
#                     )
#                     for sub_block_name, sub_block in basic_block.named_children():
#                         if sub_block_name == "downsample":
#                             torch.quantization.fuse_modules(
#                                 sub_block, [["0", "1"]], inplace=True
#                             )
#     return model
def cmap_norm(cm):
    classes = ["Basophil", "Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]
    classes = [classes[i][:4] for i in [4, 2, 3, 1, 0]]
    totals = np.sum(cm, axis=1)
    cmn = np.round(
        np.divide(cm, totals, out=np.zeros(cm.shape, dtype=float), where=totals != 0), 2
    )
    ax = heatmap(
        cmn,
        annot=cm,
        fmt="d",
        cmap="viridis",
        cbar=False,
        square=True,
        xticklabels=classes,
        yticklabels=classes,
    )
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    return ax


def fusion_regnet(model):
    for n, m in model.named_children():
        if n in ["a", "b"]:
            torch.quantization.fuse_modules(m, ["0", "1", "2"], inplace=True)
        if n in ["c"]:
            torch.quantization.fuse_modules(m, ["0", "1"], inplace=True)
        fusion_regnet(m)
    return model


def save_torchscript_model(model, model_dir, model_filename):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.jit.save(torch.jit.script(model), model_filepath)


def visualise_data(data_loader, classes, std=0, mean=0):
    plt.figure(figsize=(20, 10))
    images, labels = next(iter(data_loader))
    out = torchvision.utils.make_grid(images[0:8])
    out = out.numpy().transpose((1, 2, 0))
    if mean:
        out_pre = out * np.array(std) + np.array(mean)
        img1 = plt.imshow(np.clip(out_pre, 0, 1))
        plt.show()

    out_aug = out
    plt.figure(figsize=(20, 10))
    plt.title([classes[x] for x in labels[:8].numpy()])
    img2 = plt.imshow(np.clip(out_aug, 0, 1))
    return img2


def getStats(img_dir, samples):
    mean_sum = [0, 0, 0]
    std_sum = [0, 0, 0]
    for img in random.choices(os.listdir(img_dir), k=samples):
        data = Image.open(img_dir + img)
        mean_sum += np.mean(np.array(data) / 255, axis=(0, 1))
        std_sum += np.std(np.array(data) / 255, axis=(0, 1))
    mean_sum /= samples
    std_sum /= samples
    return mean_sum, std_sum


class CustomRegNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        base = models.regnet_y_400mf(pretrained=pretrained)
        self.stem = nn.ModuleList(base.children())[0]
        self.trunk_output = nn.ModuleList(base.children())[1]
        self.avgpool = nn.ModuleList(base.children())[2]
        torchvision.models.mobilenet_v3_small()
        # self.features = nn.Sequential(*self.features)
        # self.features.extend(nn.ModuleList(base.children())[1])
        # in_features = base.fc.in_features
        # self.fc = nn.Linear(in_features, 5)
        self.fc = base.fc
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    # def fuse_model(self):
    #     for m in self.modules():
    #         print(m)
    #         print('***************************************')
    # if type(m) == SimpleStemIN:
    #     torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
    # if type(m) == ConvNormActivation:
    #     torch.quantization.fuse_modules(m, ['0', '1'], inplace=True)
    # if type(m) == SimpleStemIN:
    #     torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
    # if type(m) == SimpleStemIN:
    #     torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
    # if type(m) == SimpleStemIN:
    #     torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)

    # Set your own forward pass
    def forward(self, x):
        x = self.quant(x)
        x = self.stem(x)
        x = self.trunk_output(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.dequant(x)

        return x


def df_to_formatted_json(df, sep="."):
    """
    The opposite of json_normalize
    """
    result = []
    for idx, row in df.iterrows():
        parsed_row = {}
        for col_label, v in row.items():
            keys = col_label.split(".")

            current = parsed_row
            for i, k in enumerate(keys):
                if i == len(keys) - 1:
                    current[k] = v
                else:
                    if k not in current.keys():
                        current[k] = {}
                    current = current[k]
        # save
        result.append(parsed_row)
    return result


def datamaker(colour, config):
    ##########################################################
    # DATA PREPARATION
    ##########################################################
    transform = transforms.Compose(
        [
            RandAugmentMC(n=config["transforms"]["n"], m=config["transforms"]["m"]),
            transforms.ToTensor(),
        ]
    )
    transform_test = transforms.Compose([transforms.ToTensor()])

    train_data = datasets.ImageFolder(
        config["data_path"] + colour + "/train", transform=transform
    )
    valid_data = datasets.ImageFolder(
        config["data_path"] + colour + "/val", transform=transform_test
    )
    test_data = datasets.ImageFolder(
        config["data_path"] + colour + "/test", transform=transform_test
    )
    validation_split = 0.9
    n_train_examples = int(len(train_data))
    n_valid_examples = len(valid_data)
    n_test_examples = len(test_data)
    train_loader = dataloader(
        train_data,
        shuffle=True,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    valid_loader = dataloader(
        valid_data,
        batch_size=config["valid_batch_size"],
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    test_loader = dataloader(
        test_data,
        batch_size=config["valid_batch_size"],
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    config["train_size"] = n_train_examples
    config["val_size"] = n_valid_examples
    print(
        "Train samples: {}    Batch size: {}".format(
            n_train_examples, len(train_loader)
        )
    )
    print(
        "Validation samples: {}     Batch size: {}".format(
            n_valid_examples, len(valid_loader)
        )
    )
    print(
        "Testing samples: {}     Batch size: {}".format(
            n_test_examples, len(test_loader)
        )
    )
    return train_loader, valid_loader, test_loader, test_data


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes, momentum=0.1),
            # Replace with ReLU
            nn.ReLU(inplace=False),
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend(
            [
                # dw
                ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup, momentum=0.1),
            ]
        )
        self.conv = nn.Sequential(*layers)
        # Replace torch.add with floatfunctional
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        if self.use_res_connect:
            return self.skip_add.add(x, self.conv(x))
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes=1000,
        width_mult=1.0,
        inverted_residual_setting=None,
        round_nearest=8,
    ):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if (
            len(inverted_residual_setting) == 0
            or len(inverted_residual_setting[0]) != 4
        ):
            raise ValueError(
                "inverted_residual_setting should be non-empty "
                "or a 4-element list, got {}".format(inverted_residual_setting)
            )

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(
            last_channel * max(1.0, width_mult), round_nearest
        )
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    block(input_channel, output_channel, stride, expand_ratio=t)
                )
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):

        x = self.quant(x)

        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        x = self.dequant(x)
        return x

    # Fuse Conv+BN and Conv+BN+Relu modules prior to quantization
    # This operation does not change the numerics
    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBNReLU:
                torch.quantization.fuse_modules(m, ["0", "1", "2"], inplace=True)
            if type(m) == InvertedResidual:
                for idx in range(len(m.conv)):
                    if type(m.conv[idx]) == nn.Conv2d:
                        torch.quantization.fuse_modules(
                            m.conv, [str(idx), str(idx + 1)], inplace=True
                        )


def onnx_export(model, loader, config):
    # Export model to ONNX
    images, _ = next(iter(train_loader))
    images = images.cpu()
    onnx_file = config["architecture"] + config["save_suffix"] + ".onnx"
    torch.onnx.export(model, images, onnx_file)
    wandb.save(onnx_file)


# Archive
##################
# with open("model_names.txt", "r") as f:
#     architecture_list = []
#     while True:
#         data = f.readline().replace("\n","")
#         architecture_list.append(data)
#         if not data:
#             break
# architecture_list = list(filter(None, architecture_list))
#################
# path = 'Data/First_Dataset/Train/'
# i=20000
# for subdir, dirs, files in os.walk(path):
#     for file in files:
#         filep = os.path.join(subdir, file)
#         filel = filep.replace('\\','/').split('/')
#         im = Image.open(filep)
#         width, height = im.size   # Get dimensions
#         new_width,new_height = (250,250)
#         left = (width - new_width)/2
#         top = (height - new_height)/2
#         right = (width + new_width)/2
#         bottom = (height + new_height)/2

#         # Crop the center of the image
#         im = im.crop((left, top, right, bottom))
#         im = im.resize((120,120))
#         black = Image.new(mode="RGB", size=(250, 250))
#         black.paste(im,(65,65))
#         black.save('Data/WBC_Classification_split/mixed_raabin/train/'+filel[3]+'/'+'img'+str(i)+'.png')
#         #os.rename(filep,'Data/WBC_mixed/'+filel[3]+'/'+filel[4]+'/'+'img'+str(i)+'.png')
#         #print('{} -> {}'.format(filel[-1],i))
#         i+=1
########################
# SEPARATE VAL,TEST,TRAIN FOLDERS TO ONE IMG/LABEL FOLDER WITH TEST/TRAIN/VAL CSV'S
#
# for colour in ["Fluor", "Green", "Purple", "Red"]:
#     for sett in ['test','val','train']:
#         img_path = 'Data/WBC_20220215/full_image/{}/{}/images'.format(colour,sett)
#         lab_path = 'Data/WBC_20220215/full_image/{}/{}/labels'.format(colour,sett)
#         img_list = os.listdir(img_path)
#         lab_list = os.listdir(lab_path)
#         with open('Data/WBC_20220215/full_image/{}/{}'.format(colour,sett) + '.csv', "w", newline='') as f:
#             writer = csv.writer(f)
#             for img in img_list:
#                 label = img[:-3]+'txt'
#                 writer.writerow([img, label])
#                 shutil.move(img_path+'/' + img, 'Data/WBC_20220215/full_image/{}/images/{}'.format(colour,img))
#                 shutil.move(lab_path+'/' + label, 'Data/WBC_20220215/full_image/{}/labels/{}'.format(colour,label))
#####################################

"""
Main file for training Yolo model on Pascal VOC and COCO dataset
"""
import copy
import json
import sys
import warnings

import torch
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.optim as optim
import torch.quantization
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from tqdm import tqdm

import config
from loss import YoloLoss
from model import YOLOv3
from utils import (cells_to_bboxes, check_class_accuracy, evaluation_box_class,
                   get_evaluation_bboxes, get_loaders, load_checkpoint,
                   mean_average_precision, plot_couple_examples,
                   save_checkpoint, size_measure, test_latency)

warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True


def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors, epoch):
    loop = tqdm(train_loader, leave=True)
    losses = []
    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE).half()
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )
        with torch.cuda.amp.autocast():
            out = model(x)
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
            )

        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update progress bar
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss, epoch=epoch)
    return mean_loss


def main():
    writer = SummaryWriter(config.RUN_NAME)
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()

    train_loader, val_loader, _ = get_loaders(
        train_csv_path=config.DATASET + "/train.csv",
        test_csv_path=config.DATASET + "/val.csv",
    )

    # config.LOAD_MODEL = False
    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE)

    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    early_counter = 0
    best_Map = 0
    for epoch in range(config.EPOCH_START, config.EPOCH_START + config.NUM_EPOCHS):
        mean_loss = train_fn(
            train_loader, model, optimizer, loss_fn, scaler, scaled_anchors, epoch
        )
        writer.add_scalar("Batch Loss", mean_loss, epoch)

        if (epoch != 0) and (epoch % 2 == 0):
            Map, precisions, class_accuracy, no_obj_accuracy, obj_accuracy = evaluate(
                model, val_loader
            )
            # Write to tensorboard
            writer.add_scalar("Class Accuracy", class_accuracy, epoch)
            writer.add_scalar("No Obj Accuracy", no_obj_accuracy, epoch)
            writer.add_scalar("Obj Accuracy", obj_accuracy, epoch)
            writer.add_scalar("MAP", Map, epoch)
            # Print results
            print(f"MAP: {Map}")
            print("Precision:")
            print(
                [
                    f"{config.CLASSES[i][:4]}: {precisions[i].item()*100:.2f}%"
                    for i in range(config.NUM_CLASSES)
                ]
            )
            print(
                f"Class accuracy is: {class_accuracy:.2f}%    No obj accuracy is: {no_obj_accuracy:.2f}%    Obj accuracy is: {obj_accuracy:.2f}%"
            )

            # Save checkpoint if MAP improves
            if (Map > best_Map) and config.SAVE_MODEL:
                save_checkpoint(model, optimizer, filename=config.CHECKPOINT_FILE)
                best_Map = Map
            else:
                early_counter += 1

            # Early stopping
            # if early_counter > 10

            model.train()
    writer.close()


def inference():
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    _, test_loader, _ = get_loaders(
        train_csv_path=config.DATASET + "/train.csv",
        test_csv_path=config.DATASET + "/val.csv",
    )
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    # load_checkpoint(config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE)
    print("=> Loading checkpoint")
    checkpoint = torch.load(config.CHECKPOINT_FILE, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    # plot_couple_examples(model, test_loader, 0.6, 0.5, scaled_anchors)

    # Measure model size
    model_size = size_measure(model)
    print("Model Size: ", model_size)

    # Measure latency
    config.BATCH_SIZE = 24
    _, test_loader, _ = get_loaders(
        train_csv_path=config.DATASET + "/train.csv",
        test_csv_path=config.DATASET + "/val.csv",
    )
    mean_latency, std_latency = test_latency(
        model, test_loader, "cuda", repetitions=500
    )
    print("Mean Latency: {} Standard Deviation: {}".format(mean_latency, std_latency))


def quantized_inference():
    ## CONFIGS
    # config.CONF_THRESHOLD = 0.1
    # config.MAP_IOU_THRESH = 0.3
    # config.NMS_IOU_THRESH = 0.3

    ## DATA
    val_loader = get_loaders("calibration", config.DATASET + "/val4.csv", batch_size=24)
    test_loader = get_loaders("test", config.DATASET + "/test4.csv", batch_size=24)
    latency_loader = get_loaders("latency", config.DATASET + "/test4.csv", batch_size=6)

    ## MODEL LOADING
    model32 = YOLOv3(num_classes=config.NUM_CLASSES)
    optimizer = optim.Adam(
        model32.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    load_checkpoint(config.CHECKPOINT_FILE, model32, optimizer, config.LEARNING_RATE)
    model8 = copy.deepcopy(model32)
    model32.eval()
    model8.eval()

    ## FUSE
    for module_name, module in model8.named_children():
        if module_name == "layers":
            for basic_block_name, basic_block in module.named_children():
                for l2n, l2 in basic_block.named_children():
                    if l2n == "conv":
                        torch.quantization.fuse_modules(
                            basic_block, [["conv", "bn", "relu"]], inplace=True
                        )
                    elif l2n == "layers":
                        for l3n, l3 in l2.named_children():
                            for l4n, l4 in l3.named_children():
                                torch.quantization.fuse_modules(
                                    l4, [["conv", "bn", "relu"]], inplace=True
                                )
    ## PREPARE
    model8.qconfig = torch.quantization.QConfig(
        activation=getattr(torch.quantization, config.OBSERVER[0]).with_args(
            dtype=torch.quint8, qscheme=getattr(torch, config.QSCHEME[0])
        ),
        weight=getattr(torch.quantization, config.OBSERVER[1]).with_args(
            dtype=torch.qint8, qscheme=getattr(torch, config.QSCHEME[1])
        ),
    )
    torch.quantization.prepare(model8, inplace=True)

    ## CALIBRATION
    model8.to("cuda")
    with torch.inference_mode():
        loop = tqdm(val_loader, leave=True)
        for batch_idx, (x, y) in enumerate(loop):
            x = x.float().to(config.DEVICE)
            model8(x)
    model8.to("cpu")

    ## CONVERSION
    print("Quantized with configuration:")
    print(config.OBSERVER, config.DTYPE, config.QSCHEME)
    model8 = torch.quantization.convert(model8, inplace=True).to("cpu")

    ## TESTING
    model32.to("cuda")
    print("Float32 Model:")
    Map, precisions, class_accuracy, no_obj_accuracy, obj_accuracy = evaluate(
        model32, test_loader, device="cuda"
    )
    print(f"MAP: {Map}")
    print("Precision:")
    print(
        [
            f"{config.CLASSES[i][:4]}: {precisions[i].item()*100:.2f}%"
            for i in range(config.NUM_CLASSES)
        ]
    )
    print(
        f"Class accuracy is: {class_accuracy:.2f}%    No obj accuracy is: {no_obj_accuracy:.2f}%    Obj accuracy is: {obj_accuracy:.2f}%"
    )
    # mean_latency, std_latency = test_latency(model32, "cpu", 6, 50)
    # print("Mean Latency: {} Standard Deviation: {}".format(mean_latency, std_latency))
    # model_size = size_measure(model32)
    # print("Model Size: ", model_size)

    # print("Int8 Model:")
    # Map, precisions, class_accuracy, no_obj_accuracy, obj_accuracy = evaluate(
    #     model8, test_loader, device="cpu"
    # )
    # print(f"MAP: {Map}")
    # print("Precision:")
    # print(
    #     [
    #         f"{config.CLASSES[i][:4]}: {precisions[i].item()*100:.2f}%"
    #         for i in range(config.NUM_CLASSES)
    #     ]
    # )
    # print(
    #     f"Class accuracy is: {class_accuracy:.2f}%    No obj accuracy is: {no_obj_accuracy:.2f}%    Obj accuracy is: {obj_accuracy:.2f}%"
    # )
    # mean_latency, std_latency = test_latency(model8, "cpu", 6, 50)
    # print("Mean Latency: {} Standard Deviation: {}".format(mean_latency, std_latency))
    # model_size = size_measure(model8)
    # print("Model Size: ", model_size)

    # plot_couple_examples(model, test_loader, 0.6, 0.5, scaled_anchors)


def evaluate(model, loader, device):
    model.eval()
    (
        pred_boxes,
        true_boxes,
        class_accuracy,
        no_obj_accuracy,
        obj_accuracy,
    ) = evaluation_box_class(
        loader,
        model,
        iou_threshold=config.NMS_IOU_THRESH,
        anchors=config.ANCHORS,
        threshold=config.CONF_THRESHOLD,
        device=device,
    )
    mapval, precisions = mean_average_precision(
        pred_boxes,
        true_boxes,
        iou_threshold=config.MAP_IOU_THRESH,
        box_format="midpoint",
        num_classes=config.NUM_CLASSES,
    )

    return mapval.item(), precisions, class_accuracy, no_obj_accuracy, obj_accuracy


def model_testing():
    # EXPORT TO ONNX TEST
    model = YOLOv3(num_classes=config.NUM_CLASSES)
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    load_checkpoint(config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE)

    dummy_input = torch.rand(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    onnx_file_name = "yolov3_test.onnx"
    torch.onnx.export(
        model, dummy_input, onnx_file_name,
    )
    # JIT TRACE TEST
    torch.jit.trace(model, dummy_input)

    torch.fx.symbolic_trace(model)


def pruning():
    """
    torch.nn.utils.prune.Identity: utility pruning method that does not prune any units but generates the pruning parametrization with a mask of ones;
    torch.nn.utils.prune.RandomUnstructured: prune (currently unpruned) entries in a tensor at random;
    torch.nn.utils.prune.L1Unstructured: prune (currently unpruned) entries in a tensor by zeroing out the ones with the lowest absolute magnitude;
    torch.nn.utils.prune.RandomStructured: prune entire (currently unpruned)rows or columns in a tensor random;
    torch.nn.utils.prune.LnStructured: prune entire (currently unpruned) rows or columns in a tensor based on their -norm (supported values of n correspond to sup-ported values for argument p in torch.norm());
    torch.nn.utils.prune.CustomFromMask: prune a tensor using a user-provided mask.
    """
    _, test_loader, _ = get_loaders(
        train_csv_path=config.DATASET + "/train.csv",
        test_csv_path=config.DATASET + "/test.csv",
    )
    config.DEVICE = "cuda"
    print("Running on ", config.DEVICE)
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE).eval()
    print("=> Loading checkpoint")
    checkpoint = torch.load(config.CHECKPOINT_FILE, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)
    anchors2 = scaled_anchors
    thresh = 0.7
    iou_thresh = 0.5
    sparcity = 0.1

    for module_name, module in model.named_children():
        if module_name == "layers":
            for basic_block_name, basic_block in module.named_children():
                for l2n, l2 in basic_block.named_children():
                    if l2n == "conv":
                        prune.l1_unstructured(l2, name="weight", amount=sparcity)
                        prune.remove(l2, "weight")
                    elif l2n == "bn":
                        pass
                    elif l2n == "relu":
                        pass
                    elif l2n == "layers":
                        for l3n, l3 in l2.named_children():
                            for l4n, l4 in l3.named_children():
                                for l5n, l5 in l4.named_children():
                                    if l5n == "conv":
                                        prune.l1_unstructured(
                                            l5, name="weight", amount=sparcity
                                        )
                                        prune.remove(l5, "weight")
    evaluate(model, test_loader)
    # repetitions = 10
    # mean_latency, std_latency = test_latency(
    #     model, test_loader, "cpu", repetitions=repetitions
    # )
    # print("Mean Latency: {} Standard Deviation: {}".format(mean_latency, std_latency))


def quantizer(upper_module_name, upper_module):
    for module_name, module in upper_module.named_children():
        if module_name == "conv":
            torch.quantization.fuse_modules(
                upper_module, [["conv", "bn", "relu"]], inplace=True
            )
        else:
            quantizer(module_name, module)


class CustomQ(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        resnet18 = torchvision.models.resnet18(pretrained=True)
        self.features = nn.ModuleList(resnet18.children())
        print(self.features)
        self.features = nn.Sequential(*self.features)

    def forward(self, input_imgs):
        output = self.features(input_imgs)


def quantized_inference_2():
    ## CONFIGS
    # config.CONF_THRESHOLD = 0.1
    # config.MAP_IOU_THRESH = 0.3
    # config.NMS_IOU_THRESH = 0.3

    ## DATA
    val_loader = get_loaders("calibration", config.DATASET + "/val.csv", batch_size=24)
    test_loader = get_loaders("test", config.DATASET + "/test.csv", batch_size=24)
    latency_loader = get_loaders("latency", config.DATASET + "/test.csv", batch_size=6)

    ## MODEL LOADING
    model32 = YOLOv3(num_classes=config.NUM_CLASSES)
    checkpoint = torch.load(config.CHECKPOINT_FILE, map_location=config.DEVICE)
    model32.load_state_dict(checkpoint["state_dict"], strict=False)
    model8 = copy.deepcopy(model32)
    model32.eval()
    model8.eval()
    # model = CustomQ()
    # print(model)
    ## FUSE
    quantizer("YoloV3", model8)
    # summary(model8.to('cuda'), input_size=(3, 512, 512))
    # ## PREPARE
    model8.qconfig = torch.quantization.QConfig(
        activation=getattr(torch.quantization, config.OBSERVER[0]).with_args(
            dtype=torch.quint8, qscheme=getattr(torch, config.QSCHEME[0])
        ),
        weight=getattr(torch.quantization, config.OBSERVER[1]).with_args(
            dtype=torch.qint8, qscheme=getattr(torch, config.QSCHEME[1])
        ),
    )
    torch.quantization.prepare(model8, inplace=True)

    # ## CALIBRATION
    model8.to("cuda")
    with torch.inference_mode():
        loop = tqdm(val_loader, leave=True)
        for batch_idx, (x, y) in enumerate(loop):
            x = x.float().to(config.DEVICE)
            model8(x)
            # if batch_idx == 5:
            #     break
    model8.to("cpu")

    # ## CONVERSION
    model8 = torch.quantization.convert(model8, inplace=True).to("cpu")
    print("Quantized with configuration:")
    print(config.OBSERVER, config.DTYPE, config.QSCHEME)

    ## TESTING
    model32.to("cuda")
    print("Float32 Model:")
    Map, precisions, class_accuracy, no_obj_accuracy, obj_accuracy = evaluate(
        model32, test_loader, device="cuda"
    )
    print(f"MAP: {Map}")
    print("Precision:")
    print(
        [
            f"{config.CLASSES[i][:4]}: {precisions[i].item()*100:.2f}%"
            for i in range(config.NUM_CLASSES)
        ]
    )
    print(
        f"Class accuracy is: {class_accuracy:.2f}%    No obj accuracy is: {no_obj_accuracy:.2f}%    Obj accuracy is: {obj_accuracy:.2f}%"
    )
    mean_latency, std_latency = test_latency(model32, "cpu", 6, 50)
    print("Mean Latency: {} Standard Deviation: {}".format(mean_latency, std_latency))
    model_size = size_measure(model32)
    print("Model Size: ", model_size)

    print("Int8 Model:")
    Map, precisions, class_accuracy, no_obj_accuracy, obj_accuracy = evaluate(
        model8, test_loader, device="cpu"
    )
    print(f"MAP: {Map}")
    print("Precision:")
    print(
        [
            f"{config.CLASSES[i][:4]}: {precisions[i].item()*100:.2f}%"
            for i in range(config.NUM_CLASSES)
        ]
    )
    print(
        f"Class accuracy is: {class_accuracy:.2f}%    No obj accuracy is: {no_obj_accuracy:.2f}%    Obj accuracy is: {obj_accuracy:.2f}%"
    )
    mean_latency, std_latency = test_latency(model8, "cpu", 6, 50)
    print("Mean Latency: {} Standard Deviation: {}".format(mean_latency, std_latency))
    model_size = size_measure(model8)
    print("Model Size: ", model_size)

    # plot_couple_examples(model, test_loader, 0.6, 0.5, scaled_anchors)


if __name__ == "__main__":
    globals()[sys.argv[1]]()

import csv
import glob
import json
import sys
import time
import warnings

import albumentations as A
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.quantization
import torchvision.transforms as transforms
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageGrab, ImageOps
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import YOLODataset
from loss import YoloLoss
from model import YOLOv3
from utils import (
    cells_to_bboxes,
    check_class_accuracy,
    evaluation_box_class,
    get_evaluation_bboxes,
    get_loaders,
    mean_average_precision,
    non_max_suppression,
    plot_couple_examples,
    save_checkpoint,
    seed_everything,
    size_measure,
    test_latency,
)

DATASET = "C:/fyp/hematology/Data/WBC_20220215/full_image/Purple/tiled"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# seed_everything()  # If you want deterministic behavior
NUM_WORKERS = 2
BATCH_SIZE = 2
IMAGE_SIZE = 512
NUM_CLASSES = 5
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.0  # 2e-4
EPOCH_START = 26
NUM_EPOCHS = 100
CONF_THRESHOLD = 0.3
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.45
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
PIN_MEMORY = True
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_FILE = "checkpoint.pth.tar"
IMG_DIR = DATASET
LABEL_DIR = DATASET

ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]  # Note these have been rescaled to be between [0, 1]

# QUANTIZATION PARAMS
OBSERVER = ["MovingAverageMinMaxObserver", "MovingAverageMinMaxObserver"]
DTYPE = ["quint8", "quint8"]
QSCHEME = [
    "per_tensor_affine",
    "per_tensor_affine",
]  # ["per_tensor_affine", "per_tensor_symmetric"]
REDUCE_RANGE = ["True", "True"]


train_transforms = A.Compose(
    [
        # A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
        # A.PadIfNeeded(
        #    min_height=int(IMAGE_SIZE * scale),
        #    min_width=int(IMAGE_SIZE * scale),
        #    border_mode=cv2.BORDER_CONSTANT,
        # ),
        # A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
        # A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
        A.ShiftScaleRotate(rotate_limit=40, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.5),
        A.Blur(p=0.05),
        A.CLAHE(p=0.05),
        A.Posterize(p=0.05),
        # A.ToGray(p=0.1),
        A.ChannelShuffle(p=0.05),
        # A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[],),
)
test_transforms = A.Compose(
    [ToTensorV2()],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)

CLASSES = ["Basophil", "Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]


def draw_box(annotations, ax, im):
    cmap = plt.get_cmap("tab20b")
    class_labels = CLASSES
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    height, width, _ = im.shape
    for box in annotations:
        assert (
            len(box) == 6
        ), "box should contain class pred, confidence, x, y, width, height. Received: {}".format(
            box
        )
        class_pred = box[0]
        conf = box[1]
        box = box[2:]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=2,
            edgecolor=colors[int(class_pred)],
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
        ax.text(
            upper_left_x * width - 1,
            upper_left_y * height - 10,
            s=f"{class_labels[int(class_pred)]} {conf*100:0.0f}%",
            color="black",
            verticalalignment="top",
            bbox={"color": colors[int(class_pred)], "pad": 0},
        )


def inference(model, anchors, img, conf=0.7, iou_thresh=0.4):
    x = img
    x = np.moveaxis(x, -1, 0)
    x = torch.from_numpy(x).unsqueeze(0)
    x = x.to("cuda").float()
    with torch.no_grad():
        out = model(x)
    bboxes = [[] for _ in range(x.shape[0])]
    for i in range(3):
        batch_size, A, S, _, _ = out[i].shape
        anchor = anchors[i]
        boxes_scale_i = cells_to_bboxes(out[i], anchor, S=S, is_preds=True)
        for idx, (box) in enumerate(boxes_scale_i):
            bboxes[idx] += box
    for i in range(batch_size):
        nms_boxes = non_max_suppression(
            bboxes[i], iou_threshold=iou_thresh, threshold=conf, box_format="midpoint",
        )
        img = cv2_box(img, nms_boxes)
    return img


def cv2_box(img, nms_boxes):
    colours = [
        (255, 0, 0),
        (0, 255, 0),
        (200, 0, 255),
        (0, 255, 255),
        (255, 255, 0),
    ]
    for box in nms_boxes:
        class_pred = box[0]
        conf = box[1]
        box = box[2:]
        width = box[2]
        height = box[3]
        start_point = (
            int((box[0] - width / 2) * img.shape[1]),
            int((box[1] - height / 2) * img.shape[0]),
        )
        end_point = (
            int((box[0] + width / 2) * img.shape[1]),
            int((box[1] + height / 2) * img.shape[0]),
        )
        img = cv2.rectangle(
            img, start_point, end_point, color=colours[int(class_pred)], thickness=3,
        )
        start_point = (start_point[0] - 5, start_point[1] - 20)
        end_point = (start_point[0] + 100, start_point[1] + 20)
        img = cv2.rectangle(
            img, start_point, end_point, color=colours[int(class_pred)], thickness=-1,
        )
        start_point = (start_point[0], start_point[1] + 15)
        font = cv2.FONT_HERSHEY_SIMPLEX
        img = cv2.putText(
            img,
            f"{CLASSES[int(class_pred)][:5]} {(conf*100):.0f}%",
            start_point,
            font,
            0.6,
            (0, 0, 0),
            2,
        )
    return img


def plot_image(image, boxes, true_boxes=None, plt_img=None):
    """Plots predicted bounding boxes on the image"""
    im = np.array(image) / 255
    height, width, _ = im.shape
    # Display the image
    if not true_boxes:
        plt_img.imshow(im)
        draw_box(boxes, plt_img, im)
    else:
        pass
    plt.show()


def main():
    model = YOLOv3(num_classes=NUM_CLASSES).to(DEVICE).eval()
    print("=> Loading checkpoint")
    checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    anchors = (
        torch.tensor(ANCHORS)
        * torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(DEVICE)
    window_pos = (100, 100, 512, 512)
    while True:
        t = time.time()
        img = np.array(
            ImageGrab.grab(
                bbox=(
                    window_pos[0],
                    window_pos[1],
                    window_pos[0] + 512,
                    window_pos[1] + 512,
                )
            )
        )
        img = inference(model, anchors, img, conf=0.8, iou_thresh=0.2)
        elapsed = time.time() - t
        fps = 1.0 / elapsed
        cv2.putText(
            img,
            f"FPS: {fps:.2f}",
            (0, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.80,
            (255, 255, 255),
            2,
        )
        cv2.imshow("window", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    globals()[sys.argv[1]]()
    ##
    # TO DO
    # -> ONNX

import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2

from utils import seed_everything

DATASET = "Data/WBC_20220215/full_image/Purple/tiled2"
# Windows: "Data/WBC_20220215/full_image/Purple/tiled2"
# Ubuntu: "Data/WBC_20220215/full_image/Purple/tiled"
# "T:/FYP_dataset/mixed"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# seed_everything()  # If you want deterministic behavior
NUM_WORKERS = 6
BATCH_SIZE = 24
IMAGE_SIZE = 512
NUM_CLASSES = 5
LEARNING_RATE = 1e-7
WEIGHT_DECAY = 1e-2  # 1e-4  # 2e-4
EPOCH_START = 0
NUM_EPOCHS = 300
CONF_THRESHOLD = 0.6
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.3
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
PIN_MEMORY = True
LOAD_MODEL = True
SAVE_MODEL = True
RUN_NAME = "baseline_m"
CHECKPOINT_FILE = RUN_NAME + ".pth.tar"
IMG_DIR = DATASET
LABEL_DIR = DATASET

ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]  # Note these have been rescaled to be between [0, 1]

# QUANTIZATION PARAMS
OBSERVER = ["MovingAverageMinMaxObserver", "MinMaxObserver"]
DTYPE = ["quint8", "quint8"]
QSCHEME = [
    "per_tensor_affine",
    "per_tensor_symmetric",
]  # ["per_tensor_affine", "per_tensor_symmetric"]


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
        A.ShiftScaleRotate(rotate_limit=20, p=0.2, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.2),
        A.Blur(p=0.05),
        # A.CLAHE(p=0.05),
        # A.Posterize(p=0.05),
        A.ToGray(p=0.95),
        # A.ChannelShuffle(p=0.05),
        # A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[],),
)
test_transforms = A.Compose(
    [
        # A.LongestMaxSize(max_size=IMAGE_SIZE),
        # A.PadIfNeeded(
        #    min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
        # ),
        # A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        # A.ToGray(p=0.95),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)

CLASSES = ["Basophil", "Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]

import os

import pandas as pd
import PIL
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from utils.augmentations import RandAugment


def dataloader(path, fold, colour, batch_size, workers, n=0, m=0):
    train_path = os.path.join(path, "train_{}.csv".format(fold))
    val_path = os.path.join(path, "val_{}.csv".format(fold))

    train_data_transform = transforms.Compose(
        [
            RandAugment(n, m),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.ToTensor(),
        ]
    )
    val_test_data_transform = transforms.Compose(
        [transforms.CenterCrop(224), transforms.ToTensor()]
    )

    train_dataset = Single_WBC_Dataset(
        csv_path=train_path, colour=colour, transform=train_data_transform
    )
    val_dataset = Single_WBC_Dataset(
        csv_path=val_path, colour=colour, transform=val_test_data_transform
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=True,
    )
    return train_dataloader, val_dataloader


def multi_cell_dataloader(
    path="./data", fold=None, batch_size=64, workers=8, n=2, m=18
):
    train_path = os.path.join(path, "train_{}.csv".format(fold))
    val_path = os.path.join(path, "val_{}.csv".format(fold))

    train_data_transform = transforms.Compose(
        [
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.ToTensor(),
        ]
    )
    train_data_transform.transforms.insert(0, RandAugment(n, m))

    val_test_data_transform = transforms.Compose(
        [transforms.CenterCrop(224), transforms.ToTensor()]
    )

    train_dataset = Multi_WBC_Dataset(
        csv_path=train_path, transform=train_data_transform
    )
    val_dataset = Multi_WBC_Dataset(
        csv_path=val_path, transform=val_test_data_transform
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers
    )

    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers
    )
    return train_dataloader, val_dataloader


class Single_WBC_Dataset(Dataset):
    def __init__(self, csv_path, colour, transform=None):
        super().__init__()
        self.transform = transform
        self.csv_path = csv_path
        self.data_csv = pd.read_csv(csv_path)
        self.colour = colour

    def __getitem__(self, index):
        label = int(self.data_csv["label"][index])
        img = PIL.Image.open(
            os.path.join(
                os.path.dirname(self.csv_path), self.data_csv[self.colour][index]
            )
        )
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data_csv)


class Multi_WBC_Dataset(Dataset):
    def __init__(self, csv_path, transform=None):
        super().__init__()
        self.transform = transform
        self.data_csv = pd.read_csv(csv_path)

    def __getitem__(self, index):
        label = int(self.data_csv["label"][index])
        Fluor_img = PIL.Image.open(self.data_csv["Fluor"][index])
        Green_img = PIL.Image.open(self.data_csv["Green"][index])
        Purple_img = PIL.Image.open(self.data_csv["Purple"][index])
        Red_img = PIL.Image.open(self.data_csv["Red"][index])
        Fluor_img = self.transform(Fluor_img)
        Green_img = self.transform(Green_img)
        Purple_img = self.transform(Purple_img)
        Red_img = self.transform(Red_img)
        return Fluor_img, Green_img, Purple_img, Red_img, label

    def __len__(self):
        return len(self.data_csv)


def main():
    print("start")


if __name__ == "__main__":
    train_data_transform = transforms.Compose(
        [
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(180),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
            ),
            transforms.ToTensor(),
        ]
    )
    csv_path = "../data/train.csv"
    multi_data = Multi_WBC_Dataset(csv_path=csv_path, transform=train_data_transform)
    B = multi_data[0][0].shape
    print(B)

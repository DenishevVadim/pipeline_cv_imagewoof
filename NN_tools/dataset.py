import os
import torch
import torchvision
import tarfile
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import time
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torchvision import transforms
from torch import optim
import matplotlib
import matplotlib.pyplot as plt
import os
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def get_train_transforms():
    return A.Compose(
        [
            A.RandomSizedCrop((140,140), width=160, height=160, p=0.25),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2,
                                     val_shift_limit=0.2, p=0.9),
                A.RandomBrightnessContrast(brightness_limit=0.2,
                                           contrast_limit=0.2, p=0.9),
            ],p=0.9),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=45, p=1),
            A.ChannelShuffle(p=0.05),
            A.FancyPCA(),
            A.GaussNoise(p=0.25),
            A.Blur(blur_limit=4,p=0.1),
            A.Cutout(num_holes=8, max_h_size=4, max_w_size=4, fill_value=0, p=0.1),
            A.Resize (160, 160,),
            A.Normalize (mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(p=1.0),
        ],
        p=1.0
    )
def get_val_transforms():
    return A.Compose(
        [
            A.Resize (160, 160,),
            A.Normalize (mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(p=1.0),
        ],
        p=1.0
    )

class CustomDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform

    def __len__(self):
        return len(self.folder)

    def __getitem__(self, idx):
        image, label = self.folder[idx]
        if self.transform is not None:
            image = np.array(image)
            image = self.transform(image=image)["image"]
        return image, label

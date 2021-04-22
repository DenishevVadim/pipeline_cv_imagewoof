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

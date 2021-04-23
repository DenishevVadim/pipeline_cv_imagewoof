import torchvision
import numpy as np
from torch.utils.data import DataLoader, Dataset
from os import listdir
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2


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
    def __init__(self, folder, transform=None, is_labeled = True):
        self.transform = transform
        self.is_labeled = is_labeled
        self.folder = folder                        # if labeled, here labels and pil.images, else - just path
        if not self.is_labeled:
            self.image_list = listdir(folder)       # list name images in folder

    def __len__(self):
        if self.is_labeled:
            return len(self.folder)                 #len for ImageFolder
        else:
            return len(self.image_list)             #len for unlabeled data

    def __getitem__(self, idx):
        if self.is_labeled:
            image, label = self.folder[idx]
            image = np.array(image)
            image = self.transform(image=image)["image"]
            return image, label
        else:
            image = cv2.imread(self.folder + '/' + self.image_list[idx])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.transform(image=image)["image"]
            return image, self.image_list[idx]      # image, image name

def get_dataset(filepath, transform, batch_size, shuffle = False, is_labeled = True):
    if not is_labeled:
        dataloader = CustomDataset(folder=filepath, transform=transform, is_labeled=is_labeled)
    else:
        folder = torchvision.datasets.ImageFolder(filepath)
        dataset = CustomDataset(folder = folder, transform = transform)
        dataloader = DataLoader(dataset, batch_size, shuffle)
    return dataloader

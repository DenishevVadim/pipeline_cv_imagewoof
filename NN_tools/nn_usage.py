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


def train_metrics(model, optimizer, scheduler, train_dl, epoch, device, grad_clip=None):
    total = 0
    sum_loss = 0
    for x, y_class in train_dl:
        lr_ = optimizer.param_groups[0]['lr']
        batch = y_class.shape[0]
        optimizer.zero_grad()
        x, y_class = x.to(device), y_class.to(device)
        out_class = model(x)
        loss = F.cross_entropy(out_class, y_class, reduction='sum')
        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_value_(model.parameters(), grad_clip)
        optimizer.step()
        total += batch
        sum_loss += loss.item()
        if epoch < 13 or epoch > 17:
            scheduler.step()
    return total, sum_loss, model, lr_


def val_metrics(model, valid_dl, device):
    start_time = time.time()
    model.eval()
    total = 0
    sum_loss = 0
    acc = 0
    for x, y_class in valid_dl:
        batch = y_class.shape[0]
        x, y_class = x.to(device), y_class.to(device)
        out_class = model(x)
        loss = F.cross_entropy(out_class, y_class, reduction='sum')

        _, preds = torch.max(out_class, dim=1)
        acc += torch.tensor(torch.sum(preds == y_class).item())
        sum_loss += loss.item()
        total += batch
    val_time = (time.time() - start_time)
    model.train()
    return sum_loss / total, acc / total, model, val_time


def fit_model(model, optimizer, scheduler, device, train_dl, val_dl, epochs, grad_clip):
    best_loss_val = 1
    for epoch in range(epochs):
        total, sum_loss, model, lr_ = train_metrics(model, optimizer, scheduler, train_dl, epoch, grad_clip, device)
        torch.cuda.empty_cache()
        sum_loss_val, acc, model, val_time = val_metrics(model, val_dl, device)
        torch.cuda.empty_cache()
        if epoch > 10 and sum_loss_val <= best_loss_val:
            best_loss_val = sum_loss_val
            torch.save(model.state_dict(), f'epoch:{epoch}_sum_loss_val:{sum_loss_val}.pt')
        print("last_lr  %.3f train_loss %.3f val_loss %.3f val_acc %.3f val_time %.3f" % (
        lr_, sum_loss / total, sum_loss_val, acc, val_time))
    return model, sum_loss / total

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import time


def train_metrics(model, optimizer, scheduler, device, train_dl, epoch, grad_clip=None):
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
    return sum_loss / total, acc / total, model, val_time


def fit_model(model, optimizer, scheduler, device, train_dl, val_dl, epochs, grad_clip):
    model.train()
    best_loss_val = 1
    for epoch in range(epochs):
        total, sum_loss, model, lr_ = train_metrics(model, optimizer, scheduler, device, train_dl, epoch, grad_clip)
        torch.cuda.empty_cache()
        sum_loss_val, acc, model, val_time = val_metrics(model, val_dl, device)
        torch.cuda.empty_cache()
        if epoch > epochs*0.5 and sum_loss_val <= best_loss_val:
            best_loss_val = sum_loss_val
            torch.save(model.state_dict(), f'epoch:{epoch}_sum_loss_val:{round(sum_loss_val,2)}.pt')
        print("last_lr  %.3f train_loss %.3f val_loss %.3f val_acc %.3f val_time %.3f" % (
        lr_, sum_loss / total, sum_loss_val, acc, val_time))
    return model, sum_loss / total


def get_predict(model, device, test_dl):
    classes = ['Australian terrier', 'Border terrier', 'Samoyed', 'Beagle', 'Shih-Tzu', 'English foxhound',
               'Rhodesian ridgeback', 'Dingo', 'Golden retriever', 'Old English sheepdog']

    model.eval()
    name_arr = []
    pred_arr = []
    preds_arr = []
    for x, name in test_dl:
        x = x.to(device)
        out_class = model(x)
        preds, pred = torch.max(out_class, dim=1)
        preds, pred = preds.detach().numpy(), pred.detach().numpy()
        name_arr = np.concatenate((name_arr, name), axis=0)
        pred = list(map(lambda x: classes[x], pred))
        pred_arr = np.concatenate((pred_arr, pred), axis=0)
        preds_arr = np.concatenate((preds_arr, preds), axis=0)
    return pd.DataFrame(data=np.array((name_arr, pred_arr, preds_arr)).T, columns=["name","class","output"])

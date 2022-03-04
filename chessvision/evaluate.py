import torch
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import confusion_matrix

import label

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_confusion_matrix(model, loader):
    model.to(device)
    y_pred = [] # save predction
    y_true = [] # save ground truth

    # iterate over data
    for x, y in loader:
        inputs, labels = x.to(device), y.to(device)
        output = model(inputs)  # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output)  # save prediction

        labels = labels.data.cpu().numpy()
        y_true.extend(labels)  # save ground truth

    return confusion_matrix(y_true, y_pred)

def plot_confusion_matrix(cf_values):
    classes = [piece.unicode_symbol() for piece in label.LABELS]
    cf_weight = cf_values / cf_values.sum(axis=1)
    plt.figure(figsize=(12, 12))
    sn.set(font_scale=1.5)
    cf = sn.heatmap(pd.DataFrame(cf_weight, index=classes, columns=classes), annot=cf_values, cbar=False, cmap="Blues")
    cf.set_xticklabels(cf.get_xmajorticklabels(), fontsize=40)
    cf.set_yticklabels(cf.get_ymajorticklabels(), fontsize=40)
    cf.set_ylabel('Ground Truth')
    cf.set_xlabel('Predicted')
    return cf.get_figure()

def accuracy(model, loader):
    model.to(device)
    correct = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(dim=1) == y).sum().item()
    
    return correct / len(loader.dataset) * 100
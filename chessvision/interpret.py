import torch
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import confusion_matrix

from .label import PIECE_LABELS


class Interpreter:
    def __init__(self, model, dataloader, classes=[piece.unicode_symbol() for piece in PIECE_LABELS]):
        self.model, self.dataloader, self.classes = model, dataloader, classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def accuracy(self):
        correct = 0
        for x, y in self.dataloader:
            x, y = x.to(self.device), y.to(self.device)
            correct += (self.model(x).argmax(dim=1) == y).sum().item()
        
        return correct / len(self.dataloader.dataset) * 100

    def top_losses(self):
        raise NotImplementedError()

    def plot_confusion_matrix(self):
        cf_values = self.confusion_matrix()
        cf_weight = cf_values / cf_values.sum(axis=1)
        plt.figure(figsize=(12, 12))
        sn.set(font_scale=1.5)
        cf = sn.heatmap(
            pd.DataFrame(cf_weight, index=self.classes, columns=self.classes), 
            annot=cf_values, cbar=False, cmap="Blues")
        cf.set_xticklabels(cf.get_xmajorticklabels(), fontsize=40)
        cf.set_yticklabels(cf.get_ymajorticklabels(), fontsize=40)
        cf.set_ylabel('Ground Truth')
        cf.set_xlabel('Predicted')
        return cf.get_figure()

    def confusion_matrix(self):
        preds = []
        labels = []

        for x, y in self.dataloader:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            preds.extend((torch.max(torch.exp(output), 1)[1]).data.cpu().numpy())
            labels.extend(y.data.cpu().numpy())

        return confusion_matrix(labels, preds)

    

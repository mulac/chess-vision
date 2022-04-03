""" For evaluating pytorch models """

import torch
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import confusion_matrix

from .label import PIECE_LABELS


class Interpreter:
    def __init__(self, model, dataloader, loss_fn, classes=[piece.unicode_symbol() for piece in PIECE_LABELS]):
        self.model, self.dataloader, self.loss_fn, self.classes = model, dataloader, loss_fn, classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.loss_fn.reduction = 'none'

    @property
    def _data(self):
        for x, y in self.dataloader:
            yield x.cpu(), y.cpu(), self.model(x.to(self.device)).cpu()

    def accuracy(self):
        correct = 0
        for _, y, o in self._data:
            correct += (o.argmax(dim=1) == y).sum().item()
        
        return correct / len(self.dataloader.dataset) * 100

    def plot_top_losses(self, shape=(3, 3)):
        losses = sorted(self.losses(), reverse=True)[:sum(shape)]
        f = plt.figure(figsize=(10, 10))
        f.suptitle("Predicted | Actual | Loss", fontsize=20)
        for i, (loss, pred, actual, img) in enumerate(losses):
            plt.subplot(shape[0], shape[1], i+1)
            plt.title(f"{self.classes[pred]} | {self.classes[actual]} | {loss:.2f}", fontsize=16)
            plt.imshow(img)
            plt.axis("off")
        return f

    def losses(self):
        for x, y, o in self._data:
            for i, loss in enumerate(self.loss_fn(o, y)):
                yield (loss, o[i].argmax(), y[i], x[i].permute(1, 2, 0))

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
        for _, y, o in self._data:
            preds.extend((torch.max(torch.exp(o), 1)[1]).data)
            labels.extend(y.data)

        return confusion_matrix(labels, preds)

    

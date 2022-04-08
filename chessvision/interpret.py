""" For evaluating pytorch models """

import torch
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import confusion_matrix

from .util import mult


class Interpreter:
    def __init__(self, model, loader, loss_fn, classes):
        self.model, self.loader, self.loss_fn, self.classes = model, loader, loss_fn, classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.loss_fn.reduction = 'none'
        self.input_shape = next(iter(self.loader))[0].shape

    @property
    def _data(self):
        """ Returns: for each sample (input, actual, prediction) """
        for x, y in self.loader:
            yield x.cpu(), y.cpu(), self.model(x.to(self.device)).cpu()

    def losses(self):
        """ Returns: for each sample (loss, pred, actual, input) """
        for x, y, o in self._data:
            for i, loss in enumerate(self.loss_fn(o, y)):
                yield loss.item(), o[i].argmax(), y[i], x[i].permute(1, 2, 0)

    def accuracy(self):
        correct = 0
        for _, y, o in self._data:
            correct += (o.argmax(dim=1) == y).sum().item()
        
        return correct / len(self.loader.dataset) * 100

    def plot_top_losses(self, shape=(3, 3)):
        losses = list(self.losses())
        _, topk = torch.topk(torch.tensor([l[0] for l in losses]), mult(shape))
        f = plt.figure(figsize=(12, 12))
        f.suptitle("Predicted | Actual | Loss", fontsize=20)
        for i, l in enumerate(topk):
            loss, pred, actual, img = losses[l]
            plt.subplot(shape[0], shape[1], i+1)
            plt.title(f"{self.classes[pred]} | {self.classes[actual]} | {loss:.2f}", fontsize=16)
            plt.imshow(img)
            plt.axis("off")
        plt.tight_layout()
        return f

    def plot_confusion_matrix(self):
        cf_values = self.confusion_matrix()
        cf_weight = cf_values / cf_values.sum(axis=1)
        plt.figure(figsize=(12, 12))
        sn.set(font_scale=1.5)
        cf = sn.heatmap(
            pd.DataFrame(cf_weight, index=self.classes, columns=self.classes), 
            annot=cf_values, cbar=False, cmap="Blues", fmt='g')
        cf.set_xticklabels(cf.get_xmajorticklabels(), fontsize=40)
        cf.set_yticklabels(cf.get_ymajorticklabels(), fontsize=40)
        cf.set_ylabel('Ground Truth')
        cf.set_xlabel('Predicted')
        plt.tight_layout()
        return cf.get_figure()

    def confusion_matrix(self):
        preds = []
        labels = []
        for _, y, o in self._data:
            preds.extend((torch.max(torch.exp(o), 1)[1]).data)
            labels.extend(y.data)

        return confusion_matrix(labels, preds)

    def filter_visual(self, layer_name, filter_id, lr=0.1, steps=30):
        """ Returns an image that maximally excites the given filter """
        layer = self.model.get_submodule(layer_name)
        def hook_fn(module, input, output): self._features = output
        hook = layer.register_forward_hook(hook_fn)
        random_image = torch.rand([1, *self.input_shape[1:]], requires_grad=True, device=self.device)
        optimizer = torch.optim.Adam([random_image], lr=lr, weight_decay=1e-8)

        for _ in range(steps):
            optimizer.zero_grad()
            self.model(random_image)
            loss = -self._features[filter_id].mean()
            loss.backward()
            optimizer.step()

        random_image.requires_grad = False
        hook.remove()
        return random_image.cpu()[0].permute(1, 2, 0)
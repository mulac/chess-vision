""" Trainer and TrainerConfig for training pytorch models on Games """

import os
import torch
import numpy as np

from typing import Tuple, Callable
from tqdm import trange
from dataclasses import dataclass
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from torchvision import transforms


from .game import Game, ChessFolder, LabelOptions, Labeller, save_games


@dataclass
class TrainerConfig:
    train_games: Tuple[Game] = (
        Game("Bird", 0, LabelOptions(flipped=True)),
        Game("Adams", 0, LabelOptions(flipped=True)),
        Game("Adams", 1),
        Game("Adams", 2),
        Game("Adams", 3),
    )
    test_games: Tuple[Game] = (
        Game("Bird", 2),
    )
    channels: int = 3
    learning_rate: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 1e-4
    epochs: int = 300
    learning_rate: float = 0.001
    momentum: float = 0.9
    epochs: int = 300
    batch_size: int = 4
    labeller: Labeller = None
    image_shape: int = None
    scheduler: _LRScheduler = None
    loss_fn: Callable = torch.nn.CrossEntropyLoss()
    train_folder: str = None
    test_folder: str = None
    transform: transforms = None
    infer_transform: transforms = None


class Trainer:
    def __init__(self, model, config, writer):
        self.model = model
        self.config = config
        self.writer = writer

        train_folder = config.train_folder if config.train_folder else save_games(config.train_games, config.labeller.label_fn, config.labeller.classes)
        test_folder = config.test_folder if config.train_folder else save_games(config.test_games, config.labeller.label_fn, config.labeller.classes)
        self.train_dataset = ChessFolder(root=train_folder, transform=config.transform)
        self.test_dataset = ChessFolder(root=test_folder, transform=config.infer_transform)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def save_checkpoint(self):
        torch.save(self.model, "model")
        torch.save(self.config, "config")

    def evaluate(self):
        loader = DataLoader(self.test_dataset, shuffle=True, batch_size=25, num_workers=2, pin_memory=True)

        losses = []
        correct = 0
        for i, (x, y) in enumerate(loader):
            x, y = x.to(self.device), y.to(self.device)
            pred = self.model(x)
            losses.append(self.config.loss_fn(pred, y).item())
            correct += (pred.argmax(dim=1) == y).sum().item()

        return losses, correct / len(loader.dataset) * 100

    def train(self):
        loss_fn = self.config.loss_fn
        optimizer = self.model.configure_optimizers(self.config)
        loader = DataLoader(self.train_dataset, shuffle=True, batch_size=self.config.batch_size, num_workers=4, pin_memory=True)
        if self.config.scheduler:
            scheduler = self.config.scheduler(optimizer, max_lr=self.config.learning_rate, steps_per_epoch=len(loader), epochs=self.config.epochs)

        def train_one_epoch():
            losses = []
            for i, (x, y) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = loss_fn(self.model(x), y)
                loss.backward()
                losses.append(loss.item())
                optimizer.step()
                if self.config.scheduler is not None: scheduler.step()
            return losses

        best_loss = float('inf')
        for epoch in (t := trange(self.config.epochs)):
            self.model.train()
            train_losses = train_one_epoch()
            self.model.eval()

            val_losses, val_accuracy = self.evaluate()

            avg_train_loss, avg_val_loss = np.mean(train_losses), np.mean(val_losses)
            t.set_description(f'LOSS: train {avg_train_loss:.3f} valid {avg_val_loss:.3f}')
            self.writer.add_scalars('LOSS', {'Training': avg_train_loss, 'Validation': avg_val_loss}, epoch + 1)
            self.writer.add_scalars('Accuracy', {'Validation': val_accuracy}, epoch + 1)

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                self.save_checkpoint()

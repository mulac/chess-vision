import logging
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from datetime import datetime

from . import models
from .game import Game
from .label import *
from .trainer import Trainer, TrainerConfig
from .interpret import Interpreter

torch.manual_seed(235235)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 40
LR = 0.00005
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 4
IMG_SIZE = 96
AUG_CROP = .9
AUG_BRIGHTNESS = .75
AUG_HUE = .1
CHANNELS = 3
LABELLER = 'occupied'
TASK_WEIGHT = 0.5

tasks = {'piece': TASK_WEIGHT, 'occupied': 1-TASK_WEIGHT}

labellers = {
    'all':      Labeller(label, ALL_LABELS, [piece.unicode_symbol() for piece in PIECE_LABELS] + ['None']),
    'piece':    Labeller(label_pieces, PIECE_LABELS, [piece.unicode_symbol() for piece in PIECE_LABELS]),
    'occupied': Labeller(label_occupied, OCCUPIED_LABELS, ["Occupied", "Empty"]),
    'color':    Labeller(label_color, COLOR_LABELS, ["White", "Black"]),
    'type':     Labeller(label_type, TYPE_LABELS, ["pawn", "knight", "bishop", "rook", "queen", "king"]),
    'board':    Labeller(label_with_board, COLOR_LABELS, ["Occupied", "Empty"])
}

config = TrainerConfig(
    # train_folder = '/tmp/chess-vision-0rbbu160',
    # test_folder = '/tmp/chess-vision-b0acgfe7',
    train_games = [
        # *(Game("Evans", i) for i in range(7)),
        Game("Adams", 1),
        Game("Adams", 2),
        Game("Adams", 3),
        Game("Kasparov", 0),
        # Game("Kasparov", 0)
    ],
    test_games = [
        Game("Evans", 7),
        Game("Bird", 2),
        Game("Kasparov", 0)
    ],
    # train_games = [Game("Nakamura", i) for i in range(3)],
    # test_games = [Game("Nakamura", 3)],
    epochs = EPOCHS,
    batch_size = BATCH_SIZE,
    learning_rate = LR,
    scheduler = torch.optim.lr_scheduler.OneCycleLR,
    weight_decay = WEIGHT_DECAY,
    momentum = MOMENTUM,
    channels = CHANNELS,
    # loss_fn=models.SingleLoss(nn.CrossEntropyLoss(torch.tensor([.6,.72,.72,.72,.72,.72,.6,.72,.72,.72,.72,.72,.16], device=device))),
    # loss_fn=models.MultiLoss(tasks, nn.ModuleDict({t: nn.CrossEntropyLoss() for t in tasks}), tasks),
    image_shape = torch.tensor((IMG_SIZE, IMG_SIZE, CHANNELS)),
    labeller = labellers[LABELLER],
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        # transforms.Grayscale(),
        # transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=AUG_BRIGHTNESS, hue=AUG_HUE),
        transforms.RandomCrop((
            int(IMG_SIZE*AUG_CROP), 
            int(IMG_SIZE*AUG_CROP))
        ),
        # transforms.RandomAffine(180),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ]),
    infer_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(IMG_SIZE)
    ])
)

if CHANNELS == 1:
    config.transform.transforms.append(transforms.Grayscale(3))

logging.info(config)

trainer = Trainer(
    models.MLP(config.image_shape, len(config.labeller.classes)),
    # models.ConvNext(config.image_shape, len(config.labeller.classes), pretrained=True),
    config,
    SummaryWriter(f"runs/chess-vision_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
    device
)

logging.info("Begin training...")
trainer.train()

logging.info("Evaluating...")
interp = Interpreter(
    model=torch.load("model"), 
    loader=DataLoader(trainer.test_dataset, batch_size=25, num_workers=2),
    loss_fn=config.loss_fn,
    labeller=config.labeller
)

logging.info(f"Accuracy: {interp.accuracy():.2f}")
trainer.writer.add_figure("Confusion Matrix", interp.plot_confusion_matrix())
trainer.writer.add_figure("Top Losses", interp.plot_top_losses((4, 4)))
trainer.writer.close()
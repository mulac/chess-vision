import logging
import torch

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from datetime import datetime

from . import models
from .game import Game
from .label import *
from .trainer import Trainer, TrainerConfig
from .interpret import Interpreter


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
LABELLER = 'type'

labellers = {
    'piece':    Labeller(PIECE_LABELS, label, [from_id(i).unicode_symbol() for i in range(len(PIECE_LABELS))]),
    'occupied': Labeller(OCCUPIED_LABELS, label_occupied, ["Occupied", "Empty"]),
    'color':    Labeller(COLOR_LABELS, label_color, ["Black", "White"]),
    'type':     Labeller(TYPE_LABELS, label_type, ["pawn", "knight", "bishop", "rook", "queen", "king"]),
}

config = TrainerConfig(
    train_folder = '/tmp/chess-vision-mm5ouwg6',
    test_folder = '/tmp/chess-vision-fby576fb',
    train_games = (
        *(Game("Evans", i) for i in range(7)),
        Game("Adams", 1),
        Game("Adams", 2),
        Game("Adams", 3)
    ),
    test_games = (
        Game("Evans", 7),
        Game("Bird", 2),
        Game("Kasparov", 0)
    ),
    epochs = EPOCHS,
    batch_size = BATCH_SIZE,
    learning_rate = LR,
    scheduler = torch.optim.lr_scheduler.OneCycleLR,
    weight_decay = WEIGHT_DECAY,
    momentum = MOMENTUM,
    channels = CHANNELS,
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

logging.info(config)

trainer = Trainer(
    models.MixModel(config.image_shape, len(config.labeller.classes), pretrained=True),
    config,
    SummaryWriter(f"runs/chess-vision_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
)

logging.info("Begin training...")
trainer.train()

logging.info("Evaluating...")
interp = Interpreter(
    model=torch.load("model"), 
    loader=DataLoader(trainer.test_dataset, batch_size=100, num_workers=4),
    loss_fn=config.loss_fn,
    classes=config.labeller.names
)

logging.info(f"Accuracy: {interp.accuracy():.2f}")
trainer.writer.add_figure("Confusion Matrix", interp.plot_confusion_matrix())
trainer.writer.add_figure("Top Losses", interp.plot_top_losses((4, 4)))
trainer.writer.close()
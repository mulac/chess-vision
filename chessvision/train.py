import torch

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from datetime import datetime

from . import models
from .game import Game
from .label import (Labeller, from_id,
    COLOR_LABELS, PIECE_LABELS, OCCUPIED_LABELS, TYPE_LABELS, 
    label, label_color, label_occupied, label_type)
from .trainer import Trainer, TrainerConfig
from .interpret import Interpreter


EPOCHS = 3
LR = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 4
IMG_SIZE = 48
AUG_CROP = .9
AUG_BRIGHTNESS = .75
AUG_HUE = .1
CHANNELS = 3
LABELLER = 'pieces'

labellers = {
    'pieces': Labeller(PIECE_LABELS, label, [from_id(i).unicode_symbol() for i in range(len(PIECE_LABELS))]),
    'occupied': Labeller(OCCUPIED_LABELS, label_occupied, ["Occupied", "Empty"]),
    'color': Labeller(COLOR_LABELS, label_color, ["White", "Black"]),
    'type': Labeller(TYPE_LABELS, label_type, ["pawn", "knight", "bishop", "rook", "queen", "king"])
}

config = TrainerConfig(
    # train_folder = '/tmp/chess-vision-2v49ovvp',
    # test_folder = '/tmp/chess-vision-5pqhxnnj',
    train_games = (
        Game("Adams", 1),
        Game("Adams", 2),
        Game("Adams", 3),
        *(Game("Evans", i) for i in range(7))
    ),
    test_games = (
        Game("Evans", 7),
        Game("Bird", 2),
        Game("Kasparov", 0)
    ),
    epochs = EPOCHS,
    batch_size = BATCH_SIZE,
    learning_rate = LR,
    weight_decay = WEIGHT_DECAY,
    momentum = MOMENTUM,
    channels = CHANNELS,
    image_shape = torch.tensor((IMG_SIZE, IMG_SIZE, CHANNELS)),
    classes = labellers[LABELLER].labels,
    label_fn = labellers[LABELLER].label_fn,
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

print(config)

trainer = Trainer(
    models.ConvRes(config.image_shape, len(config.classes), pretrained=True),
    config,
    SummaryWriter(f"runs/chess-vision_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
)

print("\nBegin training...")
trainer.train()

print("\nEvaluating...")
interp = Interpreter(
    model=torch.load("model"), 
    loader=DataLoader(trainer.test_dataset, batch_size=100, num_workers=4),
    loss_fn=config.loss_fn,
    classes=labellers[LABELLER].names
)

print(f"Accuracy: {interp.accuracy():.2f}")
trainer.writer.add_figure("Confusion Matrix", interp.plot_confusion_matrix())
trainer.writer.add_figure("Top Losses", interp.plot_top_losses((6, 6)))
trainer.writer.close()
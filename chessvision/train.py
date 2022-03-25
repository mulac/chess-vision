import os
import torch

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from datetime import datetime

from . import evaluate, models
from .label import PIECE_LABELS, OCCUPIED_LABELS, label, label_occupied
from .trainer import Trainer, TrainerConfig


EPOCHS = 300
LR = 0.001
MOMENTUM = 0.9
IMG_SIZE = 48
CHANNELS = 3
LABELLER = 'peices'

label_info = {
    'peices': (PIECE_LABELS, label),
    'occupied': (OCCUPIED_LABELS, label_occupied)
}

config = TrainerConfig(
    train_folder = '/tmp/chess-vision-3j1vgaxk',
    test_folder = '/tmp/chess-vision-75e8b4qv',
    epochs = EPOCHS,
    learning_rate = LR,
    momentum = MOMENTUM,
    channels = CHANNELS,
    image_shape = torch.tensor((IMG_SIZE, IMG_SIZE, CHANNELS)),
    classes = label_info[LABELLER][0],
    label_fn = label_info[LABELLER][1],
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        # transforms.RandomSizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ]),
    infer_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(IMG_SIZE)
    ])
)

print(config)
model = models.CNN(config.image_shape, len(config.classes))
writer = SummaryWriter(f"runs/chess-vision_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

trainer = Trainer(
    model,
    config,
    writer
)

# Train
print("\nBegin training...")
trainer.train()


# Evaluate
print("\nEvaluating...")
test_loader = DataLoader(trainer.test_dataset, batch_size=100, num_workers=4)
accuracy = evaluate.accuracy(trainer.model, test_loader)
print(f"Accuracy: {accuracy:.2f}")

writer.add_figure("Confusion Matrix", evaluate.plot_confusion_matrix(evaluate.create_confusion_matrix(model, test_loader)))
writer.close()
import torch

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from datetime import datetime

from . import models
from .label import PIECE_LABELS, OCCUPIED_LABELS, label, label_occupied
from .trainer import Trainer, TrainerConfig
from .interpret import Interpreter


EPOCHS = 3
LR = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 4
IMG_SIZE = 48
CHANNELS = 3
LABELLER = 'pieces'

label_info = {
    'pieces': (PIECE_LABELS, label),
    'occupied': (OCCUPIED_LABELS, label_occupied)
}

config = TrainerConfig(
    # train_folder = '/tmp/chess-vision-zfo1qipd',
    # test_folder = '/tmp/chess-vision-2jsnf2u7',
    epochs = EPOCHS,
    batch_size = BATCH_SIZE,
    learning_rate = LR,
    weight_decay = WEIGHT_DECAY,
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
model = models.ConvRes(config.image_shape, len(config.classes), pretrained=True)
writer = SummaryWriter(f"runs/chess-vision_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

trainer = Trainer(
    model,
    config,
    writer
)

print("\nBegin training...")
trainer.train()

print("\nEvaluating...")
interp = Interpreter(trainer.model, DataLoader(trainer.test_dataset, batch_size=100, num_workers=4), label_info[LABELLER][0])
print(f"Accuracy: {interp.accuracy():.2f}")
writer.add_figure("Confusion Matrix", interp.plot_confusion_matrix())
writer.close()
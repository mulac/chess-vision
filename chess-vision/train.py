import os

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from datetime import datetime

from trainer import Trainer, TrainerConfig
import evaluate
import models
import label


EPOCHS = 300
LR = 0.001
MOMENTUM = 0.9
IMG_SIZE = 48

config = TrainerConfig(
    # train_folder = '/tmp/chess-vision-0z_ulz3o',
    # test_folder = '/tmp/chess-vision-h77znuuu',
    epochs = EPOCHS,
    learning_rate = LR,
    momentum = MOMENTUM,
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

model = models.LeNet(3, len(label.LABELS))
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
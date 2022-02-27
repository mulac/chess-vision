import os
import torch

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import LeNet
from label import Game, LABELS, save_games


class ChessFolder(datasets.ImageFolder):
    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        return classes, {i: int(i) for i in classes}


train_games = [
#     Game("Adams", 0),
    Game("Adams", 1),
    Game("Adams", 2),
    Game("Adams", 3),
]

test_games = [
    Game("Bird", 2)
]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = LeNet(3, len(LABELS)).to(device)

data_transform = transforms.Compose([
    transforms.Resize(48),
#     transforms.Grayscale(),
#         transforms.RandomSizedCrop(224),
#         transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
])

chess_datasets = {'train': ChessFolder(root=save_games(train_games), transform=data_transform),
                  'val'  : ChessFolder(root=save_games(test_games), transform=data_transform)}

dataloaders = {'train': DataLoader(chess_datasets['train'], shuffle=True, batch_size=4, num_workers=4),
               'val': DataLoader(chess_datasets['val'], batch_size=50, num_workers=4)}

dataset_sizes = {x: len(chess_datasets[x]) for x in ['train', 'val']}

loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

EPOCHS = 300

def train_one_epoch(epoch_index):
    running_loss = 0

    for i, (x, y) in enumerate(dataloaders['train']):
        optimizer.zero_grad()

        loss = loss_fn(model(x), y)
        loss.backward()
        running_loss += loss.item()

        optimizer.step()
        
        # if i % BS == 999:
        #     last_loss = running_loss / 1000 # loss per batch
        #     print('  batch {} loss: {}'.format(i + 1, last_loss))
        #     tb_x = epoch_index * len(training_loader) + i + 1
        #     tb_writer.add_scalar('Loss/train', last_loss, tb_x)
        #     running_loss = 0.
        
    return running_loss / i


for epoch in range(EPOCHS):
    model.train(True)
    avg_loss = train_one_epoch(epoch)

    model.train(False)

    running_vloss = 0.0
    for i, vdata in enumerate(dataloaders['val']):
        vinputs, vlabels = vdata
        voutputs = model(vinputs)
        vloss = loss_fn(voutputs, vlabels)
        running_vloss += vloss
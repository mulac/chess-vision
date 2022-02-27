import os
import torch
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import trange
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix

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

print("creating datasets...")
chess_datasets = {'train': ChessFolder(root=save_games(train_games), transform=data_transform),
                  'val'  : ChessFolder(root=save_games(test_games), transform=data_transform)}

dataloaders = {'train': DataLoader(chess_datasets['train'], shuffle=True, batch_size=4, num_workers=4, pin_memory=True),
               'val': DataLoader(chess_datasets['val'], batch_size=50, num_workers=4, pin_memory=True)}

dataset_sizes = {x: len(chess_datasets[x]) for x in ['train', 'val']}

loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

writer = SummaryWriter(f"runs/chess-vision_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

EPOCHS = 300

def train_one_epoch(epoch_index):
    running_loss = 0

    for i, (x, y) in enumerate(dataloaders['train']):
        x, y = x.to(device), y.to(device)
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


print("begin training...")
for epoch in (t := trange(EPOCHS)):
    model.train(True)
    avg_loss = train_one_epoch(epoch)

    model.train(False)

    running_vloss = 0.0
    for i, (x, y) in enumerate(dataloaders['val']):
        vinputs, vlabels = x.to(device), y.to(device)
        voutputs = model(vinputs)
        vloss = loss_fn(voutputs, vlabels)
        running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    t.set_description(f'LOSS: train {avg_loss:.3f} valid {avg_vloss:.3f}')
    writer.add_scalars('LOSS', {'Training': avg_loss, 'Validation': avg_vloss}, epoch + 1)

def accuracy():
    correct = 0
    for x, y in chess_datasets['val']:
        correct += (model(x.to(device).unsqueeze(0)).argmax() == y).sum().item()
    return correct/len(chess_datasets['val'])*100

accuracy = accuracy()
print(f"Accuracy: {accuracy:.2f}")

def create_confusion_matrix(loader):
    y_pred = [] # save predction
    y_true = [] # save ground truth

    # iterate over data
    for x, y in loader:
        inputs, labels = x.to(device), y.to(device)
        output = model(inputs)  # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output)  # save prediction

        labels = labels.data.cpu().numpy()
        y_true.extend(labels)  # save ground truth

    return confusion_matrix(y_true, y_pred)

def plot_confusion_matrix(cf_values):
    classes = [piece.unicode_symbol() for piece in LABELS]
    cf_weight = cf_values / cf_values.sum(axis=1)
    plt.figure(figsize=(12, 12))
    sn.set(font_scale=1.5)
    cf = sn.heatmap(pd.DataFrame(cf_weight, index=classes, columns=classes), annot=cf_values, cbar=False, cmap="Blues")
    cf.set_xticklabels(cf.get_xmajorticklabels(), fontsize=40)
    cf.set_yticklabels(cf.get_ymajorticklabels(), fontsize=40)
    cf.set_ylabel('Ground Truth')
    cf.set_xlabel('Predicted')
    return cf.get_figure()

cf_matrix = create_confusion_matrix(dataloaders['train'])
writer.add_figure("Confusion Matrix", plot_confusion_matrix(cf_matrix), epoch)

writer.add_hparams(
    {'loss_fn': repr(loss_fn), 'optimizer': repr(optimizer), 'model': repr(model)}, 
    {'accuracy': accuracy}
)

writer.close()
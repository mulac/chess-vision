import torch
import torch.nn as nn

from functools import reduce


class MLP(nn.Module):
    def __init__(self, shape, classes, loss_fn=torch.nn.CrossEntropyLoss()):
        super(MLP, self).__init__()
        self.loss_fn = loss_fn
        self.relu = nn.ReLU()
        self.fc0 = nn.Linear(in_features=torch.prod(shape), out_features=1000)
        self.fc1 = nn.Linear(in_features=1000, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.relu(self.fc0(x))
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return self.logSoftmax(x)

    def configure_optimizers(self, config):
        return torch.optim.SGD(self.parameters(), lr=config.learning_rate, momentum=config.momentum)


class CNN(nn.Module):
    def __init__(self, shape, classes, loss_fn=torch.nn.CrossEntropyLoss()):
        super(CNN, self).__init__()
        self.loss_fn = loss_fn

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv1 = nn.Conv2d(in_channels=shape[-1], out_channels=20, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5))

        self.fc0 = nn.Linear(in_features=13520, out_features=1000)
        self.fc1 = nn.Linear(in_features=1000, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x)))
        # x = self.maxpool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc0(x))
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return self.logSoftmax(x)

    def configure_optimizers(self, config):
        return torch.optim.SGD(self.parameters(), lr=config.learning_rate, momentum=config.momentum)
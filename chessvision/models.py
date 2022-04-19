""" Various pytorch models """

import torch
import torch.nn as nn

from torchvision import models


class MLP(nn.Module):
    def __init__(self, shape, classes):
        super().__init__()
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
    def __init__(self, shape, classes):
        super().__init__()

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv1 = nn.Conv2d(in_channels=shape[-1], out_channels=20, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=12, kernel_size=(5, 5))

        self.fc0 = nn.Linear(in_features=1452, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.maxpool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc0(x))
        x = self.fc2(x)

        return self.logSoftmax(x)

    def configure_optimizers(self, config):
        return torch.optim.AdamW(self.parameters(), 
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.norm(self.conv(x))


class ConvNorm(nn.Module):
    def __init__(self, shape, classes):
        super().__init__()

        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        self.conv1 = ConvBlock(in_channels=shape[-1], out_channels=20, kernel_size=(3, 3))
        self.conv2 = ConvBlock(in_channels=20, out_channels=12, kernel_size=(3, 3))

        self.fc0 = nn.Linear(in_features=300, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=classes)
        self.dropout = nn.Dropout()
        self.logSoftmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = torch.flatten(x, 1)
        x = self.act(self.fc0(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return self.logSoftmax(x)

    def configure_optimizers(self, config):
        return torch.optim.AdamW(self.parameters(), 
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )


class ConvRes(models.ResNet):
    def __init__(self, shape, classes, pretrained=False, freeze_features=False):
        if shape[-1] != 3:
            raise ValueError("must have 3 channels")
        super().__init__(models.resnet.BasicBlock, [2, 2, 2, 2])
        if pretrained:
            state_dict = models.resnet.load_state_dict_from_url(models.resnet.model_urls['resnet18'])
            self.load_state_dict(state_dict)
        if freeze_features:
            for param in self.parameters():
                param.requires_grad = False
        self.fc = nn.Sequential(
            nn.Linear(self.fc.in_features, 64),
            nn.ReLU(),
            nn.Linear(64, classes)
        )

    def configure_optimizers(self, config):
        return torch.optim.AdamW(self.parameters(), 
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )


class MixModel(models.ResNet):
    def __init__(self, shape, classes, pretrained=False, freeze_features=False):
        if shape[-1] != 3:
            raise ValueError("must have 3 channels")
        super().__init__(models.resnet.BasicBlock, [2, 2, 2, 2])
        if pretrained:
            state_dict = models.resnet.load_state_dict_from_url(models.resnet.model_urls['resnet18'])
            self.load_state_dict(state_dict)
        if freeze_features:
            for param in self.parameters():
                param.requires_grad = False
        self.color_head = nn.Sequential(
            nn.Linear(self.fc.in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 2) # TODO don't hard code
        )
        self.piece_head = nn.Sequential(
            nn.Linear(self.fc.in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 6) # TODO don't hard code
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        color, piece = self.color_head(x), self.piece_head(x)
        white_pieces = torch.mul(color[:,0].unsqueeze(0).T, piece)
        black_pieces = torch.mul(color[:,1].unsqueeze(0).T, piece)
        return torch.cat((white_pieces, black_pieces), dim=1)

    def configure_optimizers(self, config):
        return torch.optim.AdamW(self.parameters(), 
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )


class ConvNext(models.ConvNeXt):
    def __init__(self, shape, classes, pretrained=False, freeze_features=False):
        if shape[-1] != 3:
            raise ValueError("must have 3 channels")
        super().__init__([
            models.convnext.CNBlockConfig(96, 192, 3),
            models.convnext.CNBlockConfig(192, 384, 3),
            models.convnext.CNBlockConfig(384, 768, 9),
            models.convnext.CNBlockConfig(768, None, 3),
        ], 0.1)
        if pretrained:
            state_dict = models.convnext.load_state_dict_from_url(models.convnext._MODELS_URLS['convnext_tiny'])
            self.load_state_dict(state_dict)
        if freeze_features:
            for param in self.parameters():
                param.requires_grad = False
        self.classifier = nn.Sequential(
            models.convnext.LayerNorm2d(768, eps=1e-6), 
            nn.Flatten(1), 
            nn.Linear(768, classes)
        )

    def configure_optimizers(self, config):
        return torch.optim.AdamW(self.parameters(), 
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
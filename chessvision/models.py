""" Various pytorch models """

import copy
import torch
import torch.nn as nn

from torchvision import models


# ======
# LOSSES 
# ======

class SingleLoss(nn.Module):
    def __init__(self, loss_fn):
        super().__init__()
        self.loss_fn = loss_fn
    
    def disable_reduction(self):
        self.loss_fn.reduction = 'none'
    
    def forward(self, pred, target):
        return {'total': self.loss_fn(pred, target)}


class MultiLoss(nn.modules.loss._Loss):
    def __init__(self, tasks: list, loss_funcs: nn.ModuleDict, loss_weights: dict):
        super().__init__()
        assert(set(tasks) == set(loss_funcs.keys()))
        assert(set(tasks) == set(loss_weights.keys()))
        self.tasks = tasks
        self.loss_funcs = loss_funcs
        self.loss_weights = loss_weights

    def disable_reduction(self):
        for loss_fn in self.loss_funcs.values():
            loss_fn.reduction = 'none'

    def forward(self, pred, target):
        targets = {'color': torch.where(target < 6, 1, 0), 'piece': torch.where(target < 6, target, target - 6)}
        losses = {task: self.loss_funcs[task](pred[task], targets[task]) for task in self.tasks}
        losses['total'] = torch.sum(torch.stack([self.loss_weights[t] * losses[t] for t in self.tasks]))
        return losses


# ======
# MODELS
# ======

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
        self.pool = nn.AdaptiveMaxPool2d(5)

        self.conv1 = ConvBlock(in_channels=shape[-1], out_channels=20, kernel_size=(11, 11))
        self.conv2 = ConvBlock(in_channels=20, out_channels=12, kernel_size=(3, 3))

        self.fc0 = nn.Linear(in_features=300, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=classes)
        self.dropout = nn.Dropout()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(self.conv2(x))
        x = torch.flatten(x, 1)
        x = self.act(self.fc0(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def configure_optimizers(self, config):
        return torch.optim.AdamW(self.parameters(), 
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )


class ColorNet(nn.Module):
    def __init__(self, shape, classes):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=shape[-1], out_channels=48, kernel_size=11),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(48),
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(start_dim=1)  ,
            nn.Linear(in_features=25600, out_features=120),
            nn.ReLU(),
            nn.BatchNorm1d(120),
            nn.Linear(in_features=120, out_features=60),
            nn.ReLU(),
            nn.Linear(in_features=60, out_features=classes)
        )
        
    def forward(self, x):
        return self.net(x)

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


class DualNet(models.ResNet):
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

    def backbone(self, x):
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

        return x

    def forward(self, x):
        im, sq = x
        im = self.backbone(im)
        sq = self.backbone(sq)
        combined = torch.cat((im.view(im.size(0), -1), sq.view(sq.size(0), -1)), dim=1)
        
        return self.fc(combined)

    def configure_optimizers(self, config):
        return torch.optim.AdamW(self.parameters(), 
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )


class MultiModel(models.ResNet):
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
        self.color_head = nn.Linear(self.fc.in_features, 2)
        self.piece_head = nn.Linear(self.fc.in_features, 6)

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
        
        return {'color': self.color_head(x), 'piece': self.piece_head(x)}

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
        self.color_head = nn.Linear(self.fc.in_features, 2)
        self.piece_head = nn.Linear(self.fc.in_features, 6)

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
        
        c, p = self.color_head(x), self.piece_head(x)
        white_pieces = torch.mul(c[:,0].unsqueeze(0).T, p)
        black_pieces = torch.mul(c[:,1].unsqueeze(0).T, p)
        return torch.cat((white_pieces, black_pieces), dim=1)

    def configure_optimizers(self, config):
        return torch.optim.AdamW(self.parameters(), 
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

class AllModel(models.ResNet):
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
        self.occupancy_head = nn.Sequential(
            nn.Linear(self.fc.in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # TODO don't hard code
        )
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
        self.softmax = nn.Softmax()

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

        color, piece, occupied = self.color_head(x), self.piece_head(x), self.occupancy_head(x)
        white_pieces = torch.mul(color[:,0].unsqueeze(0).T, piece)
        black_pieces = torch.mul(color[:,1].unsqueeze(0).T, piece)
        pieces = torch.cat((white_pieces, black_pieces), dim=1)
        pieces = torch.mul(pieces, 1 - occupied)
        pieces = torch.cat((pieces, occupied), dim=1)
        return self.softmax(pieces)

    def configure_optimizers(self, config):
        return torch.optim.AdamW(self.parameters(), 
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
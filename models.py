# Max-Heinrich Laves
# Institute of Mechatronic Systems
# Leibniz Universität Hannover, Germany
# 2019

import torch
import torchvision


class BaselineResNet(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self._resnet = torchvision.models.resnet18(pretrained=True)
        self._resnet.fc = torch.nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x, dropout=False, p=0.5):
        y = self._resnet(x)
        return y


class BayesianResNet(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self._resnet = torchvision.models.resnet18(pretrained=True)
        self._resnet.fc = torch.nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x, dropout=False, p=0.5):
        x = self._resnet.conv1(x)
        x = self._resnet.bn1(x)
        x = self._resnet.relu(x)
        x = self._resnet.maxpool(x)

        if dropout:
            x = torch.nn.functional.dropout(x, p=p, training=True, inplace=False)
        x = self._resnet.layer1(x)

        if dropout:
            x = torch.nn.functional.dropout(x, p=p, training=True, inplace=False)
        x = self._resnet.layer2(x)

        if dropout:
            x = torch.nn.functional.dropout(x, p=p, training=True, inplace=False)
        x = self._resnet.layer3(x)

        if dropout:
            x = torch.nn.functional.dropout(x, p=p, training=True, inplace=False)
        x = self._resnet.layer4(x)

        x = self._resnet.avgpool(x)
        x = x.view(x.size(0), -1)

        if dropout:
            x = torch.nn.functional.dropout(x, p=p, training=True, inplace=False)
        y = self._resnet.fc(x)

        return y

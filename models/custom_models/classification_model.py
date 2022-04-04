import torch
import torch.nn as nn


class ClassificationMLPModel(nn.Module):
    def __init__(self, features, layer_1, layer_2, classes):
        super().__init__()
        self.fc1 = nn.Linear(features, layer_1)
        self.relu_fc1 = torch.nn.ReLU()
        self.fc2 = nn.Linear(layer_1, layer_2)
        self.relu_fc2 = torch.nn.ReLU()
        self.fc3 = nn.Linear(layer_2, classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.fc2(x)
        x = self.relu_fc2(x)
        x = self.fc3(x)
        return x


class ClassificationZeroHModel(nn.Module):
    def __init__(self, features, classes):
        super().__init__()
        self.fc1 = nn.Linear(features, classes)

    def forward(self, x):
        x = self.fc1(x)
        return x

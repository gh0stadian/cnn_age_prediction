import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineModel(nn.Module):
    def __init__(self, classification_model, l_1_size=8, l_2_size=16, l_3_size=32, l_4_size=64, kernel_size=(3, 3)):
        super().__init__()
        self.conv1 = nn.Conv2d(3, l_1_size, kernel_size, padding='same')
        self.conv2 = nn.Conv2d(l_1_size, l_2_size, kernel_size, padding='same')
        self.conv3 = nn.Conv2d(l_2_size, l_3_size, kernel_size, padding='same')
        self.conv4 = nn.Conv2d(l_3_size, l_4_size, kernel_size, padding='same')
        self.pool = nn.MaxPool2d(2, 2)
        self.classification_model = classification_model

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.classification_model(x)
        return x


class ConvModel(nn.Module):
    def __init__(self, conv_layers, in_channels=3, kernel_sizes=None, lin_layers=None, num_classes=1):
        super().__init__()

        self.conv_layers = nn.ModuleList()
        self.lin_layers = nn.ModuleList()

        if kernel_sizes is None:
            kernel_sizes = [(5, 5)] * len(conv_layers)

        in_size = in_channels
        for layer_size, kernel_size in zip(conv_layers, kernel_sizes):
            self.conv_layers.append(nn.Conv2d(in_size, layer_size, kernel_size, padding='same'))
            in_size = layer_size

        in_features = int(((224 / pow(2, len(conv_layers))) ** 2) * conv_layers[-1])
        for layer_size in lin_layers:
            self.lin_layers.append(nn.Linear(in_features, layer_size))
            in_features = layer_size

        self.out_layer = nn.Linear(in_features, num_classes)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = torch.nn.LeakyReLU()

    def forward(self, x):
        for layer in self.conv_layers:
            x = self.pool(F.relu(layer(x)))

        x = torch.flatten(x, 1)

        for layer in self.lin_layers:
            x = self.relu(layer(x))
        x = self.out_layer(x)
        return x

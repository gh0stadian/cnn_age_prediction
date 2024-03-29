import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveModel(nn.Module):
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

        self.adaptive = nn.AdaptiveAvgPool2d(output_size=(7, 7))

        # in_features = 12544
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
        x = self.adaptive(x)

        x = torch.flatten(x, 1)

        for layer in self.lin_layers:
            x = self.relu(layer(x))
        x = self.out_layer(x)
        return x

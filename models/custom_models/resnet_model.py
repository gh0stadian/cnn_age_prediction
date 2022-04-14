import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=(1, 1)):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1, stride=stride, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1, stride=stride, bias=False)

        if self.in_channels != self.out_channels:
            self.downsample = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(1, 1), stride=(1, 1))

        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        if self.in_channels != self.out_channels:
            identity = self.downsample(identity)

        x = self.relu(self.conv1(x))
        x = self.conv2(x)

        x += identity
        x = self.relu(x)
        return x


class ResBatchNormBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=(1, 1)):
        super(ResBatchNormBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        if self.in_channels != self.out_channels:
            self.downsample = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(1, 1), stride=(1, 1))

        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        if self.in_channels != self.out_channels:
            identity = self.downsample(identity)

        x = self.relu(self.batch_norm2(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, Block, channel_list, num_classes=1, linear_layers=None):
        super(ResNet, self).__init__()
        in_features = 64
        self.lin_layers = nn.ModuleList()

        self.conv1 = nn.Conv2d(3, in_features, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(in_features)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1)

        self.res_blocks = nn.ModuleList()

        for out_features in channel_list:
            self.res_blocks.append(Block(in_features, out_features, (1, 1)))
            in_features = out_features

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        in_features = channel_list[-1]

        if linear_layers is not None:
            for layer_size in linear_layers:
                self.lin_layers.append(nn.Linear(in_features, layer_size))
                in_features = layer_size

        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))

        for res_block in self.res_blocks:
            x = self.max_pool(x)
            x = res_block(x)

        x = self.avg_pool(x)
        x = x.reshape(x.shape[0], -1)

        for layer in self.lin_layers:
            x = layer(x)
        x = self.fc(x)

        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=(1, 1)):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=(1, 1), stride=(1, 1),
                               padding=0
                               )
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))

        x = self.relu(self.batch_norm2(self.conv2(x)))

        x = self.conv3(x)
        x = self.batch_norm3(x)

        # downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        # add identity
        x += identity
        x = self.relu(x)

        return x

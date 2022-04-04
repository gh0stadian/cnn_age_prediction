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

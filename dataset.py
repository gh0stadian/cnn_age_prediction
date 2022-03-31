import numpy as np
from PIL import Image
import os
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms


class FaceDataset(Dataset):
    def __init__(self, dataframe, img_root_dir, num_classes, transform=None):
        self.dataframe = dataframe
        self.img_root_dir = img_root_dir
        self.num_classes = num_classes
        self.transform = transform
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img = self.get_img(idx)
        age = F.one_hot(torch.tensor(self.dataframe['age'][idx]), num_classes=self.num_classes).to(dtype=torch.float32)
        name = str(self.dataframe['name'][idx])

        if self.transform:
            img = self.transform(img)

        return img, age, name

    def get_img(self, idx):
        img_path = os.path.join(self.img_root_dir, self.dataframe['full_path'][idx])
        img = Image.open(img_path).convert('RGB')
        img = torchvision.transforms.ToTensor()(img)
        return img.to(dtype=torch.float32)

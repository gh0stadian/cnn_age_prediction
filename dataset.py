import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms.functional as fn


class FaceDataset(Dataset):
    def __init__(self, dataframe, img_root_dir, transform=None):
        self.dataframe = dataframe
        self.img_root_dir = img_root_dir
        self.transform = transform
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img = self.get_img(idx)
        age = torch.tensor(self.dataframe['age'][idx]).to(self.device),
        name = str(self.dataframe['name'][idx])

        if self.transform:
            img = self.transform(img)

        return img, age, name

    def get_img(self, idx):
        img_path = os.path.join(self.img_root_dir, self.dataframe['full_path'][idx])
        img = read_image(img_path).to(self.device)
        return img

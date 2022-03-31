import pandas as pd
import torch
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from torch.utils.data import DataLoader, random_split

from models.pretrained_model import get_pretrained_model
from torchvision import transforms
from dataset import FaceDataset
from train import Model

# one image is approx. 440MB
wandb.init(
    project="test-project",
    entity="jbdb",
    config={
        'epoch': 100,
        'batch_size': 10,
        'lr': 0.01,
    }
)
wandb_logger = WandbLogger(log_model=True)

df = pd.read_csv("cured_imdb.csv", low_memory=False)
num_classes = df['age'].max() + 1

dataset = FaceDataset(dataframe=df,
                      img_root_dir='imdb_crop/',
                      num_classes=num_classes,
                      transform=transforms.Compose([transforms.Resize((224, 224))])
                      )

valid_test_length = round(len(dataset) * 0.15)
lengths = [len(dataset) - 2 * valid_test_length, valid_test_length, valid_test_length]
train_dataset, valid_dataset, test_dataset = random_split(dataset, lengths)

train_loader = DataLoader(train_dataset, batch_size=wandb.config['batch_size'], shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=wandb.config['batch_size'], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=wandb.config['batch_size'], shuffle=False)

model = get_pretrained_model(num_classes=num_classes)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config['lr'])

model = Model(model, criterion, optimizer)

checkpoint_callback = ModelCheckpoint(dirpath="checkpoints/", save_top_k=2, monitor="val/loss")
early_stop_callback = EarlyStopping(monitor="val/loss", patience=3, mode="min")
trainer = pl.Trainer(gpus=1,
                     benchmark=True,
                     logger=wandb_logger,
                     max_epochs=1,
                     devices=1,
                     callbacks=[checkpoint_callback, early_stop_callback]
                     )
trainer.fit(model, train_loader, valid_loader)

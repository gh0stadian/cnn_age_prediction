import pandas as pd
import torch
import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader, random_split

from config import wandb, criterion, img_transforms, model, optimizer, scheduler
from dataset import FaceDataset
from train import PLModel

wandb_logger = WandbLogger(log_model=True)

df = pd.read_csv(wandb.config['dataset_path'], low_memory=False)

dataset = FaceDataset(dataframe=df,
                      img_root_dir=wandb.config['img_root_dir'],
                      num_classes=1,
                      transform=img_transforms
                      )

# TODO split csv files instead
valid_test_length = round(len(dataset) * 0.2)
lengths = [len(dataset) - 2 * valid_test_length, valid_test_length, valid_test_length]
train_dataset, valid_dataset, test_dataset = random_split(dataset, lengths)

train_loader = DataLoader(train_dataset, batch_size=wandb.config['batch_size'], shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=wandb.config['batch_size'], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=wandb.config['batch_size'], shuffle=False)

model = PLModel(model, criterion, optimizer, scheduler)

checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{wandb.config['checkpoint']}/",
                                      save_top_k=2,
                                      monitor="val/loss"
                                      )

early_stop_callback = EarlyStopping(monitor="val/loss", patience=wandb.config['es_patience'], mode="min")

trainer = pl.Trainer(gpus=1,
                     benchmark=True,
                     logger=wandb_logger,
                     max_epochs=wandb.config['epoch'],
                     callbacks=[checkpoint_callback, early_stop_callback]
                     )

trainer.fit(model, train_loader, valid_loader)
trainer.test(ckpt_path="best")
wandb.finish()

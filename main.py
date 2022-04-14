import pandas as pd
import torch
import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader, random_split

from config import wandb, criterion, img_transforms, model, optimizer, test_df, train_df, test_log_df, scheduler
from dataset import FaceDataset
from train import PLModel

wandb_logger = WandbLogger(log_model=True)

train_dataset = FaceDataset(dataframe=train_df,
                            img_root_dir=wandb.config['img_root_dir'],
                            num_classes=1,
                            transform=img_transforms
                            )

test_dataset = FaceDataset(dataframe=test_df,
                           img_root_dir=wandb.config['img_root_dir'],
                           num_classes=1,
                           transform=img_transforms
                           )

log_dataset = FaceDataset(dataframe=test_log_df,
                          img_root_dir=wandb.config['img_root_dir'],
                          num_classes=1,
                          transform=img_transformsa
                          )

valid_test_length = round(len(train_dataset) * 0.2)
lengths = [len(train_dataset) - valid_test_length, valid_test_length]
train_dataset, valid_dataset = random_split(train_dataset, lengths)

train_loader = DataLoader(train_dataset, batch_size=wandb.config['batch_size'], shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=wandb.config['batch_size'], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=wandb.config['batch_size'], shuffle=False)
log_loader = DataLoader(log_dataset, batch_size=wandb.config['batch_size'], shuffle=False)

model = PLModel(model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler)

wandb_logger.watch(model)

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
trainer.test(dataloaders=test_loader)
trainer.test(dataloaders=log_loader)
wandb.finish()

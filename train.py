import torch
from config import wandb
import pytorch_lightning as pl


class PLModel(pl.LightningModule):
    def __init__(self, model, criterion, optimizer, scheduler=None):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def configure_optimizers(self):
        if not self.scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                   mode='min',
                                                                   factor=wandb.config['scheduler_factor'],
                                                                   patience=wandb.config['scheduler_patience'],
                                                                   min_lr=1e-6,
                                                                   verbose=True
                                                                   )
            return {'optimizer': self.optimizer, 'lr_scheduler': self.scheduler, 'monitor': 'val/loss'}

        else:
            return self.optimizer

    def on_epoch_start(self):
        self.log('lr', self.optimizer.param_groups[0]['lr'])

    def training_step(self, batch, batch_idx):
        img, y_true, name = batch
        y_pred = self.model.forward(img)
        y_true = y_true.unsqueeze(1)
        loss = self.criterion(y_pred, y_true)
        self.log('train/loss', loss)
        self.log('train/mae', self.calculate_mae(y_pred, y_true))
        return loss

    def validation_step(self, batch, batch_idx):
        img, y_true, name = batch
        y_pred = self.model.forward(img)
        y_true = y_true.unsqueeze(1)
        loss = self.criterion(y_pred, y_true)
        self.log('val/loss', loss)
        self.log('val/mae', self.calculate_mae(y_pred, y_true))
        self.log('val/rmse', self.calculate_rmse(y_pred, y_true))
        return loss

    def calculate_mae(self, y_pred, y_true):
        mae = torch.abs(y_pred - y_true).float().mean()
        return mae

    def calculate_rmse(self, y_pred, y_true):
        diff = torch.square(y_pred) - torch.square(y_true)
        rmse = torch.sqrt(torch.abs(diff)).float().mean()
        return rmse

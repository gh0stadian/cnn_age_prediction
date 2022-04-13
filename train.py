import torch
from config import wandb
import pytorch_lightning as pl
import wandb as w_and_b


class PLModel(pl.LightningModule):
    def __init__(self, model, criterion, optimizer, scheduler=None):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def configure_optimizers(self):
        if self.scheduler is not None:
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
        self.log('train/rmse', self.calculate_rmse(y_pred, y_true))
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

    def test_step(self, batch, batch_idx):
        img, y_true, name = batch
        if len(img) <= 10 and batch_idx == 0:
            for i in zip(img, y_true, name):
                image, label, actor_name = i
                y_pred = self.model.forward(image[None, :])
                log_img = [
                    w_and_b.Image(image, caption=f"Predicted: {y_pred.item()}, True: {label}, name: {actor_name}")]
                self.logger.log_image(key="images", images=log_img)
        else:
            y_pred = self.model.forward(img)
            y_true = y_true.unsqueeze(1)
            loss = self.criterion(y_pred, y_true)
            self.log('test/loss', loss)
            self.log('test/mae', self.calculate_mae(y_pred, y_true))
            self.log('test/rmse', self.calculate_rmse(y_pred, y_true))
            return loss

    def calculate_mae(self, y_pred, y_true):
        mae = torch.abs(y_pred - y_true).float().mean()
        return mae

    def calculate_rmse(self, y_pred, y_true):
        diff = torch.square(y_pred) - torch.square(y_true)
        rmse = torch.sqrt(torch.abs(diff)).float().mean()
        return rmse

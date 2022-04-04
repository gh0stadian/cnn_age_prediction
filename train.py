import torch
import pytorch_lightning as pl


class PLModel(pl.LightningModule):
    def __init__(self, model, criterion, optimizer):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def configure_optimizers(self):
        return self.optimizer

    def training_step(self, batch, batch_idx):
        img, y_true, name = batch
        y_pred = self.model.forward(img)
        loss = self.criterion(y_pred, y_true)
        self.log('train/loss', loss)
        self.log('train/mae', self.calculate_mae(y_pred, y_true))
        return loss

    def validation_step(self, batch, batch_idx):
        img, y_true, name = batch
        y_pred = self.model.forward(img)
        loss = self.criterion(y_pred, y_true)
        self.log('val/loss', loss)
        self.log('val/mae', self.calculate_mae(y_pred, y_true))
        self.log('val/rmse', self.calculate_rmse(y_pred, y_true))

    def calculate_mae(self, y_pred, y_true):
        # for categorical only
        # _, y_pred_indices = y_pred.topk(1)
        # _, y_true_indices = y_true.topk(1)
        mae = torch.abs(y_pred - y_true).float().mean()
        return mae

    def calculate_rmse(self, y_pred, y_true):
        # for categorical only
        # _, y_pred_indices = y_pred.topk(1)
        # _, y_true_indices = y_true.topk(1)
        rmse = torch.sqrt(torch.square(y_pred) - torch.square(y_true)).float().mean()
        return rmse

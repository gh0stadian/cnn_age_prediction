import pytorch_lightning as pl


class Model(pl.LightningModule):
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
        return loss

    def validation_step(self, batch, batch_idx):
        img, y_true, name = batch
        y_pred = self.model.forward(img)
        loss = self.criterion(y_pred, y_true)
        self.log('val/loss', loss)

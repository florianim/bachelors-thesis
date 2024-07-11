import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import R2Score
from sklearn.metrics import median_absolute_error

class PriceFNN(pl.LightningModule):
    def __init__(self, config):
        super(PriceFNN, self).__init__()
        self.input_size = config["input_size"]
        self.l1_size = config["l1_size"]
        self.l2_size = config["l2_size"]
        self.output_size = config["output_size"]
        self.learning_rate = config["learning_rate"]
        self.do = config["do"]

        self.layer_1 = nn.Linear(self.input_size, self.l1_size)
        self.layer_2 = nn.Linear(self.l1_size, self.l2_size)
        self.output = nn.Linear(self.l2_size, self.output_size)

        self.dropout = nn.Dropout(self.do)

        self.relu = nn.ReLU()     

    def forward(self, x):
        out = self.layer_1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.layer_2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.output(out)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        mse_loss = nn.MSELoss()
        y_hat = self(x)
        loss = mse_loss(y_hat, y)
        self.log("loss", loss, on_step=False, on_epoch=True)
        r2 = R2Score()
        self.log("train_r2", r2(y_hat, y), on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        mse_loss = nn.MSELoss()
        y_hat = self(x)
        loss = mse_loss(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return {"y_hat": y_hat, "y": y}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def custom_histogram_adder(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)
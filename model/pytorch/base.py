from type import PytorchConfig
import pandas as pd
import pytorch_lightning as pl
from model.base import Model


class PytorchModel(Model):
    def __init__(self, id: str, config: PytorchConfig):
        self.config = config
        self.id = id

    def preload(self):
        pass

    def fit(self, dataset: pd.DataFrame) -> None:
        trainer = pl.Trainer(gpus=4, num_nodes=8, precision=16, limit_train_batches=0.5)
        trainer.fit(model, train_loader, val_loader)

    def predict(self, dataset: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def is_fitted(self) -> bool:
        raise NotImplementedError()


import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from .decoder import Decoder


class PytorchBase(pl.LightningModule, Model):
    def __init__(self, id: str, config: PytorchConfig):
        self.config = config
        self.id = id
        self.model = Decoder(config)

    def preload(self):
        pass

    def fit(self, dataset: pd.DataFrame) -> None:
        trainer = pl.Trainer(gpus=4, num_nodes=8, precision=16, limit_train_batches=0.5)
        trainer.fit(self.model, train_loader, val_loader)

    def predict(self, dataset: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def is_fitted(self) -> bool:
        raise NotImplementedError()

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28)
        )

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("val_loss", loss)


# data
dataset = MNIST("", train=True, download=True, transform=transforms.ToTensor())
mnist_train, mnist_val = random_split(dataset, [55000, 5000])

train_loader = DataLoader(mnist_train, batch_size=32)
val_loader = DataLoader(mnist_val, batch_size=32)

# model
model = LitAutoEncoder()

# training
trainer = pl.Trainer(gpus=4, num_nodes=8, precision=16, limit_train_batches=0.5)
trainer.fit(model, train_loader, val_loader)

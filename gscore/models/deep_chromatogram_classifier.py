import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F


class DeepChromModel(pl.LightningModule):

    def __init__(self, learning_rate=0.0005):

        self.lr = learning_rate

        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=10,
                kernel_size=5,
                padding="same",
            ), nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=1
            ),
            nn.Conv2d(
                10, 42,
                kernel_size=5,
                padding="same",
            ), nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            nn.BatchNorm2d(42),
            nn.Flatten()
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(1008, 1008), nn.ReLU(),
            nn.BatchNorm1d(1008),
            nn.Linear(1008, 1008), nn.ReLU(),
            nn.Linear(1008, 1)
        )

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=(self.lr or self.learning_rate)
        )

        scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-6)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train_loss"
        }

    def training_step(self, batch, batch_idx):

        chromatograms, labels = batch

        y_hat = self(chromatograms)

        loss = F.binary_cross_entropy_with_logits(
            y_hat,
            labels
        )

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )

        return loss

    def validation_step(self, batch, batch_idx):

        chromatograms, labels = batch

        y_hat = self(chromatograms)

        loss = F.binary_cross_entropy_with_logits(
            y_hat,
            labels
        )

        return loss

    def test_step(self, batch, batch_idx):

        chromatograms, labels = batch

        y_hat = self(chromatograms)

        loss = F.binary_cross_entropy_with_logits(
            y_hat,
            labels
        )

        labels_hat = torch.argmax(y_hat, dim=1)

        accuracy = torch.sum(labels_hat == labels).item() / (len(labels) * 1.0)

        self.log_dict({
            'test_loss': loss,
            'test_acc': accuracy,
        })

    def predict_step(self, batch, batch_idx, dataloader_idx=0):

        chromatograms, = batch

        return self(chromatograms)

    def forward(self, chromatogram):

        out = self.conv_layers(chromatogram)

        out = self.linear_layers(out)

        return out

from typing import Dict, Any, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import torchvision.models as models

from gps.models.base_model import Scorer


class DeepChromScorer(Scorer):
    def __init__(
        self,
        max_epochs: int = 1000,
        gpus: int = 1,
        threads: int = 1,
        initial_lr: float = 0.005,
        early_stopping: int = 10,
        training: bool = True,
        embedding: bool = False,
    ):

        self.model = DeepChromModel(learning_rate=initial_lr, embedding=embedding)

        ###TODO:
        ### set trainer base dir so that the checkpoints are not everywhere.
        ### Pass in by CLI

        if gpus > 0:

            self.trainer = Trainer(
                max_epochs=max_epochs,
                gpus=gpus,
                callbacks=[
                    LearningRateMonitor(logging_interval="epoch"),
                    EarlyStopping(
                        monitor="train_loss",
                        min_delta=0.00001,
                        patience=early_stopping,
                        verbose=True,
                        mode="min",
                    ),
                ],
                auto_select_gpus=True,
            )

        else:

            self.trainer = Trainer(
                max_epochs=max_epochs,
                gpus=gpus,
                callbacks=[
                    LearningRateMonitor(logging_interval="epoch"),
                    EarlyStopping(
                        monitor="train_loss",
                        min_delta=0.00,
                        patience=early_stopping,
                        verbose=True,
                        mode="min",
                    ),
                ],
            )

        self.threads = threads
        self.gpus = gpus
        self.initial_lr = initial_lr
        self.training = training
        self.embedding = embedding

    def fit(self, data: np.ndarray, labels: np.ndarray) -> None:

        chromatograms = torch.from_numpy(data).type(torch.FloatTensor)
        labels = torch.from_numpy(labels).type(torch.FloatTensor)

        chromatogram_dataset = TensorDataset(chromatograms, labels)

        train_length = int(0.9 * len(chromatogram_dataset))

        validation_length = len(chromatogram_dataset) - train_length

        train_dataset, validation_dataset = torch.utils.data.random_split(
            chromatogram_dataset, (train_length, validation_length)
        )

        chromatogram_dataloader = DataLoader(
            train_dataset, batch_size=100, shuffle=True, num_workers=self.threads
        )

        validation_dataloader = DataLoader(
            validation_dataset, batch_size=100, num_workers=10
        )

        self.trainer.fit(
            self.model,
            train_dataloaders=chromatogram_dataloader,
            val_dataloaders=validation_dataloader,
        )

    def score(self, data: np.ndarray) -> np.ndarray:

        if self.gpus > 0:

            trainer = Trainer(gpus=self.gpus, auto_select_gpus=True)

        else:

            trainer = Trainer(gpus=self.gpus)

        chromatograms = torch.from_numpy(data).type(torch.FloatTensor)

        prediction_dataloader = DataLoader(
            TensorDataset(chromatograms), num_workers=self.threads, batch_size=10000
        )

        predictions = trainer.predict(self.model, dataloaders=prediction_dataloader)

        predictions = torch.cat(predictions, 0)

        predictions_array: np.ndarray = predictions.numpy()

        return predictions_array

    def predict(self, data: np.ndarray) -> np.ndarray:

        if self.gpus > 0:

            trainer = Trainer(gpus=self.gpus, auto_select_gpus=True)

        else:

            trainer = Trainer(gpus=self.gpus)

        chromatograms = torch.from_numpy(data).type(torch.FloatTensor)

        prediction_dataloader = DataLoader(
            TensorDataset(chromatograms), num_workers=self.threads, batch_size=10000
        )

        predictions = trainer.predict(self.model, dataloaders=prediction_dataloader)

        probabilities = torch.cat(predictions, 0)

        probabilities = torch.sigmoid(probabilities)

        classes = torch.where(probabilities > 0.5, 1.0, 0.0)

        return classes.numpy()

    def encode(self, data: np.ndarray) -> np.ndarray:

        if self.gpus > 0:

            trainer = Trainer(gpus=self.gpus, auto_select_gpus=True)

        else:

            trainer = Trainer(gpus=self.gpus)

        chromatograms = torch.from_numpy(data).type(torch.FloatTensor)

        prediction_dataloader = DataLoader(
            TensorDataset(chromatograms), num_workers=self.threads, batch_size=10000
        )

        self.model.embedding = True

        embeddings = trainer.predict(self.model, dataloaders=prediction_dataloader)

        combined_embeddings = torch.cat(embeddings, 0)

        self.model.embedding = False

        return np.array(combined_embeddings.numpy(), dtype=np.float64)

    def probability(self, data: np.ndarray) -> np.ndarray:

        if self.gpus > 0:

            trainer = Trainer(gpus=self.gpus, auto_select_gpus=True)

        else:

            trainer = Trainer(gpus=self.gpus)

        chromatograms = torch.from_numpy(data).type(torch.FloatTensor)

        prediction_dataloader = DataLoader(
            TensorDataset(chromatograms), num_workers=self.threads, batch_size=10000
        )

        predictions = trainer.predict(self.model, dataloaders=prediction_dataloader)

        probabilities = torch.cat(predictions, 0)

        probabilities = torch.sigmoid(probabilities.to(dtype=torch.float64)).numpy()

        return probabilities

    def predict_proba(self, data: np.ndarray) -> np.ndarray:
        return self.probability(data)

    def save(self, model_path: str = "") -> None:
        self.trainer.save_checkpoint(model_path)

    def load(self, model_path: str) -> None:
        self.model = DeepChromModel.load_from_checkpoint(
            checkpoint_path=model_path,
            learning_rate=self.initial_lr,
            training=self.training,
            embedding=self.embedding,
        )

    def evaluate(self, data: np.ndarray, labels: np.ndarray) -> float:

        predictions = self.probability(data)

        return float(roc_auc_score(labels, predictions))


class DeepChromModel(pl.LightningModule):  # type: ignore
    def __init__(self, learning_rate: float = 0.0005, embedding: bool = False):

        self.lr = learning_rate
        self.embedding = embedding

        super().__init__()

        self.input_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=3,
                kernel_size=1,
                padding="same",
            ),
            nn.BatchNorm2d(3),
            nn.ReLU(),
        )

        backbone = models.resnet18(pretrained=False)

        num_filters = backbone.fc.in_features

        layers = list(backbone.children())[:-1]

        self.resnet = nn.Sequential(*layers)

        self.encoder = nn.Sequential(
            nn.Linear(num_filters, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 50),
        )

        self.output_net = nn.Sequential(nn.Linear(50, 50), nn.ReLU(), nn.Linear(50, 1))

        self._init_params()

    def _init_params(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.Adam(
            self.parameters(), lr=(self.lr or self.learning_rate)
        )

        scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-6)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train_loss",
        }

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:

        chromatograms, labels = batch

        y_hat = self(chromatograms)

        loss = F.binary_cross_entropy_with_logits(y_hat, labels)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        chromatograms, labels = batch

        y_hat = self(chromatograms)

        loss = F.binary_cross_entropy_with_logits(y_hat, labels)

        return loss

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        chromatograms, labels = batch

        y_hat = self(chromatograms)

        loss = F.binary_cross_entropy_with_logits(y_hat, labels)

        labels_hat = torch.argmax(y_hat, dim=1)

        accuracy = torch.sum(labels_hat == labels).item() / (len(labels) * 1.0)

        self.log_dict(
            {
                "test_loss": loss,
                "test_acc": accuracy,
            }
        )

    def predict_step(
        self,
        batch: Tuple[
            torch.Tensor,
        ],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:

        (chromatograms,) = batch

        if self.embedding:

            out = self.input_conv(chromatograms)

            out = self.resnet(out).flatten(1)

            out = self.encoder(out)

        else:

            out = self(chromatograms)

        return out

    def forward(self, chromatogram: torch.Tensor) -> torch.Tensor:

        out = self.input_conv(chromatogram)

        out = self.resnet(out).flatten(1)

        out = self.encoder(out)

        out = self.output_net(out)

        return out

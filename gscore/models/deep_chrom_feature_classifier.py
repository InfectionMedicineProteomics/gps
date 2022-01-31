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

from gscore.models.base_model import Scorer


class DeepChromFeatureScorer(Scorer):

    def __init__(self, max_epochs: int = 1000, gpus: int = 1, threads: int = 1, initial_lr=0.05, num_features=0):

        self.model = DeepChromFeatureModel(
            learning_rate=initial_lr,
            num_features=num_features
        )

        ###TODO:
        ### set trainer base dir so that the checkpoints are not everywhere.
        ### Pass in by CLI

        self.trainer = Trainer(
            max_epochs=max_epochs,
            gpus=gpus,
            callbacks=[
                LearningRateMonitor(logging_interval="epoch"),
                EarlyStopping(
                    monitor="train_loss",
                    min_delta=0.00,
                    patience=10,
                    verbose=False,
                    mode="min"
                )
            ]
        )
        self.threads = threads
        self.gpus = gpus
        self.num_features = num_features

    def fit(self, chromatograms: np.ndarray, scores: np.ndarray, labels) -> None:

        chromatograms = torch.from_numpy(chromatograms).type(torch.FloatTensor)
        scores = torch.from_numpy(scores).type(torch.FloatTensor)
        labels = torch.from_numpy(labels).type(torch.FloatTensor)

        chromatogram_dataset = TensorDataset(
            chromatograms,
            scores,
            labels
        )

        train_length = int(0.9 * len(chromatogram_dataset))

        validation_length = len(chromatogram_dataset) - train_length

        train_dataset, validation_dataset = torch.utils.data.random_split(
            chromatogram_dataset,
            (train_length, validation_length)
        )

        chromatogram_dataloader = DataLoader(
            train_dataset,
            batch_size=100,
            shuffle=True,
            num_workers=self.threads
        )

        validation_dataloader = DataLoader(
            validation_dataset,
            batch_size=100,
            num_workers=10
        )

        self.trainer.fit(
            self.model,
            train_dataloaders=chromatogram_dataloader,
            val_dataloaders=validation_dataloader
        )

    def score(self, chromatograms: np.ndarray, scores: np.ndarray) -> np.ndarray:
        trainer = Trainer(
            gpus=self.gpus
        )

        chromatograms = torch.from_numpy(
            chromatograms
        ).type(torch.FloatTensor)

        scores = torch.from_numpy(
            scores
        ).type(torch.FloatTensor)

        prediction_dataloader = DataLoader(
            TensorDataset(chromatograms, scores),
            num_workers=self.threads,
            batch_size=10000
        )

        predictions = trainer.predict(
            self.model,
            dataloaders=prediction_dataloader
        )

        predictions = torch.cat(predictions, 0)

        probabilities = torch.sigmoid(predictions).numpy()

        # Set probabilities that equal 1.0 to the next highest probability in the array for stable logit transforms
        probabilities[probabilities == 1.0] = probabilities[probabilities < 1.0].max()

        predictions = np.log(probabilities / (1.0 - probabilities))

        return predictions

    def probability(self, chromatograms: np.ndarray, scores: np.ndarray) -> np.ndarray:

        return torch.sigmoid(
            torch.from_numpy(self.score(chromatograms, scores)).type(torch.FloatTensor)
        ).numpy()

    def predict_proba(self, chromatograms: np.ndarray, scores: np.ndarray) -> np.ndarray:
        return self.probability(chromatograms, scores)

    def evaluate(self, chromatograms: np.ndarray, scores: np.ndarray, labels: np.ndarray) -> float:

        probabilities = self.predict_proba(
            chromatograms,
            scores
        )

        return roc_auc_score(labels, probabilities)

    def save(self, model_path: str = ""):

        self.trainer.save_checkpoint(
            model_path
        )

    def load(self, model_path: str):

        self.model = DeepChromFeatureModel.load_from_checkpoint(
            checkpoint_path=model_path,
            num_features=self.num_features
        )


class DeepChromFeatureModel(pl.LightningModule):

    def __init__(self, learning_rate=0.005, num_features=0):
        self.lr = learning_rate

        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=10,
                kernel_size=5,
                padding="same",
            ),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=1
            ),
            nn.Conv2d(
                10, 42,
                kernel_size=5,
                padding="same",
            ),
            nn.BatchNorm2d(42),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            nn.Flatten()
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(1008 + num_features, 1008), nn.ReLU(),
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

        chromatograms, scores, labels = batch

        y_hat = self(chromatograms, scores)

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
        chromatograms, scores, labels = batch

        y_hat = self(chromatograms, scores)

        loss = F.binary_cross_entropy_with_logits(
            y_hat,
            labels
        )

        return loss

    def test_step(self, batch, batch_idx):
        chromatograms, scores, labels = batch

        y_hat = self(chromatograms, scores)

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

        chromatograms, scores = batch

        return self(chromatograms, scores)

    def forward(self, chromatogram, scores):

        out = self.conv_layers(chromatogram)

        out = torch.cat(
            (
                out,
                scores
            ),
            dim=1
        )

        out = self.linear_layers(out)

        return out

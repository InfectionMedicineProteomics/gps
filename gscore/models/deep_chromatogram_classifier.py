import numpy as np
import pytorch_lightning as pl
import torch  # type: ignore
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from torch import nn  # type: ignore
from torch.optim.lr_scheduler import ReduceLROnPlateau  # type: ignore
from torch.nn import functional as F  # type: ignore
from torch.utils.data import TensorDataset, DataLoader  # type: ignore

from gscore.models.base_model import Scorer

from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler  # type: ignore

from sklearn.pipeline import Pipeline  # type: ignore


class DeepChromScorer(Scorer):
    def __init__(
        self, max_epochs: int = 1000, gpus: int = 1, threads: int = 1, initial_lr=0.005, early_stopping=25,
            training=True
    ):
        super().__init__()
        self.model = DeepChromModel(learning_rate=initial_lr, training=training)

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
                        min_delta=0.00,
                        patience=early_stopping,
                        verbose=True,
                        mode="min",
                    ),
                ],
                auto_select_gpus=True
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

    def fit(self, data, labels) -> None:

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

        #probabilities = torch.sigmoid(predictions.to(dtype=torch.float64)).numpy()

        # transform = Pipeline(
        #     [
        #         ("robust_scaler", StandardScaler()),
        #         ("min_max_scaler", MinMaxScaler(feature_range=(-1, 1))
        #          )
        #     ]
        # )

        #probabilities = torch.sigmoid(probabilities.to(dtype=torch.float64)).numpy()

        #probabilities[probabilities == 1.0] = probabilities[probabilities < 1.0].max()

        #scores = transform.fit_transform(predictions.numpy())

        return predictions.numpy()

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

    def save(self, model_path: str = ""):
        self.trainer.save_checkpoint(model_path)

    def load(self, model_path: str):
        self.model = DeepChromModel.load_from_checkpoint(checkpoint_path=model_path)


class InceptionBlock(nn.Module):
    """
    Inputs:
        c_in - Number of input feature maps from the previous layers
        c_red - Dictionary with keys "3x3" and "5x5" specifying the output of the dimensionality reducing 1x1 convolutions
        c_out - Dictionary with keys "1x1", "3x3", "5x5", and "max"
        act_fn - Activation class constructor (e.g. nn.ReLU)
    """

    def __init__(self, c_in, c_red: dict, c_out):

        super().__init__()

        # 1x1 convolution branch
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(c_in, c_out["1x1"], kernel_size=1),
            nn.BatchNorm2d(c_out["1x1"]),
            nn.ReLU()
        )

        # 3x3 convolution branch
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(c_in, c_red["3x3"], kernel_size=1),
            nn.BatchNorm2d(c_red["3x3"]),
            nn.ReLU(),
            nn.Conv2d(c_red["3x3"], c_out["3x3"], kernel_size=3, padding=1),
            nn.BatchNorm2d(c_out["3x3"]),
            nn.ReLU()
        )

        # 5x5 convolution branch
        self.conv_5x5 = nn.Sequential(
            nn.Conv2d(c_in, c_red["5x5"], kernel_size=1),
            nn.BatchNorm2d(c_red["5x5"]),
            nn.ReLU(),
            nn.Conv2d(c_red["5x5"], c_out["5x5"], kernel_size=5, padding=2),
            nn.BatchNorm2d(c_out["5x5"]),
            nn.ReLU()
        )

        # Max-pool branch
        self.max_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            nn.Conv2d(c_in, c_out["max"], kernel_size=1),
            nn.BatchNorm2d(c_out["max"]),
            nn.ReLU()
        )

    def forward(self, x):
        x_1x1 = self.conv_1x1(x)
        x_3x3 = self.conv_3x3(x)
        x_5x5 = self.conv_5x5(x)
        x_max = self.max_pool(x)
        x_out = torch.cat([x_1x1, x_3x3, x_5x5, x_max], dim=1)
        return x_out


import torchvision.models as models

class DeepChromModel(pl.LightningModule):
    def __init__(self, learning_rate=0.0005, training=False):
        self.lr = learning_rate

        super().__init__()

        self.input_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=3,
                kernel_size=1,
                padding="same",
            ),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )

        backbone = models.resnet18(pretrained=False)

        num_filters = backbone.fc.in_features

        layers = list(backbone.children())[:-1]

        self.feature_extractor = nn.Sequential(*layers)

        self.output_net = nn.Linear(num_filters, 1)

        # self.inception_blocks = nn.Sequential(
        #     InceptionBlock(
        #         64,
        #         c_red={
        #             "3x3": 32,
        #             "5x5": 16
        #         },
        #         c_out={
        #             "1x1": 16,
        #             "3x3": 32,
        #             "5x5": 8,
        #             "max": 8
        #         }
        #     ),
        #     InceptionBlock(
        #         64,
        #         c_red={"3x3": 32, "5x5": 16},
        #         c_out={"1x1": 24, "3x3": 48, "5x5": 12, "max": 12}
        #     ),
        #     nn.MaxPool2d(3, stride=2, padding=1),  # 32x32 => 16x16
        #     InceptionBlock(
        #         96,
        #         c_red={"3x3": 32, "5x5": 16},
        #         c_out={"1x1": 24, "3x3": 48, "5x5": 12, "max": 12}
        #     ),
        #     InceptionBlock(
        #         96,
        #         c_red={"3x3": 32, "5x5": 16},
        #         c_out={"1x1": 16, "3x3": 48, "5x5": 16, "max": 16}
        #     ),
        #     InceptionBlock(
        #         96,
        #         c_red={"3x3": 32, "5x5": 16},
        #         c_out={"1x1": 16, "3x3": 48, "5x5": 16, "max": 16}
        #     ),
        #     InceptionBlock(
        #         96,
        #         c_red={"3x3": 32, "5x5": 16},
        #         c_out={"1x1": 32, "3x3": 48, "5x5": 24, "max": 24}
        #     ),
        #     nn.MaxPool2d(3, stride=2, padding=1),  # 16x16 => 8x8
        #     InceptionBlock(
        #         128,
        #         c_red={"3x3": 48, "5x5": 16},
        #         c_out={"1x1": 32, "3x3": 64, "5x5": 16, "max": 16}
        #     ),
        #     InceptionBlock(
        #         128,
        #         c_red={"3x3": 48, "5x5": 16},
        #         c_out={"1x1": 32, "3x3": 64, "5x5": 16, "max": 16}
        #     ),
        # )
        #
        # self.output_net = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Flatten(),
        #     nn.Linear(128, 1)
        # )

        self._init_params()

        # self.conv_layers = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=1,
        #         out_channels=10,
        #         kernel_size=5,
        #         padding="same",
        #     ),
        #     nn.BatchNorm2d(10),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=1),
        #     nn.Conv2d(
        #         10,
        #         42,
        #         kernel_size=5,
        #         padding="same",
        #     ),
        #     nn.BatchNorm2d(42),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Flatten(),
        # )
        #
        # self.linear_layers = nn.Sequential(
        #     nn.Linear(1008, 1008),
        #     nn.ReLU(),
        #     nn.Linear(1008, 1008),
        #     nn.ReLU(),
        #     nn.Linear(1008, 1),
        # )

    def _init_params(self):
        # Based on our discussion in Tutorial 4, we should initialize the
        # convolutions according to the activation function
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=(self.lr or self.learning_rate)
        )

        scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-6)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train_loss",
        }

    def training_step(self, batch, batch_idx):

        chromatograms, labels = batch

        y_hat = self(chromatograms)

        loss = F.binary_cross_entropy_with_logits(y_hat, labels)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        chromatograms, labels = batch

        y_hat = self(chromatograms)

        loss = F.binary_cross_entropy_with_logits(y_hat, labels)

        return loss

    def test_step(self, batch, batch_idx):
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

    def predict_step(self, batch, batch_idx, dataloader_idx=0):

        (chromatograms,) = batch

        return self(chromatograms)

    def forward(self, chromatogram):

        # out = self.conv_layers(chromatogram)
        #
        # out = self.linear_layers(out)

        out = self.input_conv(chromatogram)

        out = self.feature_extractor(out).flatten(1)

        out = self.output_net(out)

        return out

from joblib import dump, load  # type: ignore

from sklearn.linear_model import SGDClassifier  # type: ignore
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.metrics import roc_auc_score  # type: ignore


from imblearn.ensemble import BalancedBaggingClassifier, EasyEnsembleClassifier  # type: ignore
from torch.utils.data import Subset, DataLoader

from xgboost import XGBClassifier

from torch.nn import functional as F
from torch import nn

import numpy as np

import torch

MODELS = {"adaboost": AdaBoostClassifier}


class Scorer:
    def fit(self, data: np.ndarray, labels: np.ndarray):

        self.model.fit(data, labels)

    def probability(self, data: np.ndarray) -> np.ndarray:

        return self.model.predict_proba(data)[:, 1]

    def predict_proba(self, data: np.ndarray) -> np.ndarray:

        return self.model.predict_proba(data)[:, 1]

    def score(self, data: np.ndarray) -> np.ndarray:

        probabilities = self.model.predict_proba(data)[:, 1]

        return np.log(probabilities / (1 - probabilities))

    def save(self, model_path: str):

        dump(self.model, model_path)

    def load(self, model_path: str):

        self.model = load(model_path)

    def evaluate(self, data: np.ndarray, labels: np.ndarray) -> float:

        probabilities = self.probability(data)

        return roc_auc_score(labels, probabilities)


class SGDScorer(Scorer):

    model: SGDClassifier

    def __init__(self, class_weights: np.ndarray):

        self.model = SGDClassifier(
            alpha=1e-05,
            average=True,
            loss="log",
            max_iter=500,
            penalty="l2",
            shuffle=True,
            tol=0.0001,
            learning_rate="adaptive",
            eta0=0.001,
            fit_intercept=True,
            random_state=42,
            class_weight=dict(enumerate(class_weights)),
        )


class XGBoostScorer(Scorer):

    model: XGBClassifier

    def __init__(self, scale_pos_weight: float):

        self.model = XGBClassifier(
            n_estimators=100,
            verbosity=1,
            objective="binary:logistic",
            n_jobs=10,
            random_state=42,
            scale_pos_weight=scale_pos_weight,
        )


class EasyEnsembleScorer(Scorer):

    model: EasyEnsembleClassifier
    submodel: SGDClassifier

    def __init__(self, class_weights: np.ndarray):

        self.submodel = SGDClassifier(
            alpha=1e-05,
            average=True,
            loss="log",
            max_iter=500,
            penalty="l2",
            shuffle=True,
            tol=0.0001,
            learning_rate="adaptive",
            eta0=0.001,
            fit_intercept=True,
            random_state=42,
            class_weight=dict(enumerate(class_weights)),
        )

        self.model = EasyEnsembleClassifier(
            base_estimator=self.submodel,
            n_estimators=100,
            sampling_strategy="auto",
            random_state=42,
            n_jobs=10,
            verbose=True,
        )


class XGBEnsembleScorer(Scorer):

    model: EasyEnsembleClassifier
    submodel: SGDClassifier

    def __init__(self, scale_pos_weight: float):

        self.submodel = XGBClassifier(
            n_estimators=10,
            verbosity=1,
            objective="binary:logistic",
            n_jobs=5,
            random_state=42,
            scale_pos_weight=scale_pos_weight,
        )

        self.model = EasyEnsembleClassifier(
            base_estimator=self.submodel,
            n_estimators=10,
            sampling_strategy="auto",
            random_state=42,
            n_jobs=2,
            verbose=True,
        )


class BalancedBaggingScorer(Scorer):

    model: BalancedBaggingClassifier
    submodel: SGDClassifier

    def __init__(self, class_weights: np.ndarray):

        self.submodel = SGDClassifier(
            alpha=1e-05,
            average=True,
            loss="log",
            max_iter=500,
            penalty="l2",
            shuffle=True,
            tol=0.0001,
            learning_rate="adaptive",
            eta0=0.001,
            fit_intercept=True,
            random_state=42,
            class_weight=dict(enumerate(class_weights)),
        )

        self.model = BalancedBaggingClassifier(
            base_estimator=self.submodel,
            n_estimators=100,
            bootstrap=True,
            sampling_strategy="auto",
            random_state=42,
            n_jobs=10,
            verbose=True,
        )


class GradientBoostingScorer(Scorer):

    model: GradientBoostingClassifier

    def __init__(self):

        self.model = GradientBoostingClassifier()


class AdaBoostSGDScorer(Scorer):

    model: AdaBoostClassifier
    submodel: SGDClassifier

    def __init__(self, class_weights: np.ndarray):

        self.submodel = SGDClassifier(
            alpha=1e-05,
            average=True,
            loss="log",
            max_iter=500,
            penalty="l2",
            shuffle=True,
            tol=0.0001,
            learning_rate="adaptive",
            eta0=0.001,
            fit_intercept=True,
            random_state=42,
            class_weight=dict(enumerate(class_weights)),
        )

        self.model = AdaBoostClassifier(
            base_estimator=self.submodel,
            n_estimators=100,
            learning_rate=1.0,
            algorithm="SAMME.R",
            random_state=42,
        )


class ChromatogramModel:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_losses = []
        self.val_losses = []

    def train_step(self, chroms, peakgroups, labels):

        self.optimizer.zero_grad()

        yhat = self.model(chroms, peakgroups)

        loss = self.criterion(yhat.reshape(-1), labels.double())

        loss.backward()

        self.optimizer.step()

        return loss.item()

    def train_val_split(self, dataset, val_split=0.10):

        train_idx, val_idx = train_test_split(
            list(range(len(dataset))), test_size=val_split
        )

        training_set = Subset(dataset, train_idx)
        validation_set = Subset(dataset, val_idx)

        return training_set, validation_set

    def validation_step(self, chroms, peakgroups, labels):

        yhat = self.model(chroms, peakgroups)

        val_loss = self.criterion(yhat.reshape(-1), labels)

        return val_loss.item()

    def load(self, saved_model_path):

        self.model.load_state_dict(torch.load(saved_model_path))

    def eval_test_accuracy(self, testing_dataset):

        testing_loader = DataLoader(
            testing_dataset, batch_size=32, num_workers=5, drop_last=True
        )

        accuracies = []

        for i, (peakgroups, chroms, labels) in enumerate(testing_loader):

            peakgroups = peakgroups.to(self.device)
            chroms = chroms.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                predictions = self.model.predict(chroms.double(), peakgroups)

                accuracy = (
                    (predictions.detach() == labels.reshape((-1, 1)).detach()).sum()
                    / labels.shape[0]
                ).item()

                accuracies.append(accuracy)

            accuracies.append(accuracy)

        return np.mean(accuracies)

    def train(self, training_data, val_split=0.10, batch_size=32, n_epochs=50):

        training_split, validation_split = self.train_val_split(
            training_data, val_split
        )

        training_loader = DataLoader(
            training_split, batch_size=batch_size, shuffle=True, num_workers=5
        )

        validation_loader = DataLoader(
            validation_split, batch_size=batch_size, shuffle=True, num_workers=5
        )

        try:

            for epoch in range(1, n_epochs + 1):

                losses = []
                validation_losses = []

                accuracies = []

                for i, (peakgroups, chroms, labels) in enumerate(training_loader):

                    peakgroups = peakgroups.to(self.device)
                    chroms = chroms.to(self.device)
                    labels = labels.to(self.device)

                    self.optimizer.zero_grad()

                    yhat = self.model(chroms, peakgroups)

                    loss = self.criterion(yhat.reshape(-1), labels.double())

                    loss.backward()

                    self.optimizer.step()

                    losses.append(loss.item())

                    percentage = (i / len(training_loader)) * 100.0

                    print("Epoch percentage: ", percentage, end="\r")

                training_loss = np.mean(losses)

                losses.append(training_loss)

                with torch.no_grad():

                    val_losses = []

                    for i, (peakgroups, chroms, labels) in enumerate(validation_loader):

                        peakgroups = peakgroups.to(self.device)
                        chroms = chroms.to(self.device)
                        labels = labels.to(self.device)

                        val_loss = self.validation_step(chroms, peakgroups, labels)

                        val_losses.append(val_loss)

                        predictions = self.model.predict(chroms.double(), peakgroups)

                        accuracy = (
                            (
                                predictions.detach() == labels.reshape((-1, 1)).detach()
                            ).sum()
                            / predictions.shape[0]
                        ).item()

                        accuracies.append(accuracy)

                    batch_val_loss = np.mean(val_losses)

                    validation_losses.append(val_loss)

                epoch_loss = np.mean(losses)
                epoch_val_accuracy = np.mean(accuracies)
                epoch_val_loss = np.mean(validation_losses)

                self.train_losses.append(epoch_loss)
                self.val_losses.append(epoch_val_loss)

                print(
                    f"epoch {epoch}, loss {epoch_loss}, val loss {epoch_val_loss}, val accuracy {epoch_val_accuracy}"
                )

        except Exception as e:

            torch.save(self.model.state_dict(), "./chrom.pth")

            raise e

        torch.save(self.model.state_dict(), "./chrom.pth")


class ChromatogramProbabilityModel(nn.Module):
    def __init__(self, n_features, sequence_length):

        super().__init__()

        self.conv1d = nn.Conv1d(
            in_channels=n_features,
            out_channels=7,
            kernel_size=(3,),
            stride=(1,),
            padding="same",
        )

        self.n_features = n_features
        self.sequence_length = sequence_length

        self.layer_dim = 2
        self.hidden_dim = 20

        self.rnn = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=self.layer_dim,
            batch_first=True,
            dropout=0.2,
            bidirectional=True,
        )

        self.linear = nn.Linear((2 * self.hidden_dim) * sequence_length + 3, 42)
        self.linear_2 = nn.Linear(42, 42)
        self.linear_3 = nn.Linear(42, 42)
        self.linear_4 = nn.Linear(42, 1)

    def forward(self, chromatogram, peakgroup):

        batch_size, seq_len, _ = chromatogram.size()

        out = self.conv1d(chromatogram.double())

        out = out.permute(0, 2, 1)

        out, _ = self.rnn(out.double())

        out = out.contiguous().view(batch_size, -1)

        out = torch.cat((out, peakgroup), 1)

        out = self.linear(out)

        out = F.relu(out)

        out = self.linear_2(out)

        out = F.relu(out)

        out = self.linear_3(out)

        out = F.relu(out)

        out = self.linear_4(out)

        return out

    def predict(self, data, peakgroup):

        out = self.forward(data, peakgroup)

        probabilities = torch.sigmoid(out)

        return (probabilities > 0.5).double()

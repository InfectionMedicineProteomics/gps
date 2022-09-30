import numpy as np
from joblib import dump, load
from sklearn.metrics import roc_auc_score


class Scorer:
    def fit(self, data: np.ndarray, labels: np.ndarray) -> None:

        self.model.fit(data, labels)

    def probability(self, data: np.ndarray) -> np.ndarray:

        probabilities = self.model.predict_proba(data)[:, 1]

        probabilities = 1 / (1 + np.exp(-probabilities))

        return probabilities

    def predict_proba(self, data: np.ndarray) -> np.ndarray:

        probabilities = self.model.predict_proba(data)[:, 1]

        probabilities = 1 / (1 + np.exp(-probabilities))

        return probabilities

    def score(self, data: np.ndarray) -> np.ndarray:

        return self.model.predict_proba(data)[:, 1]

    def save(self, model_path: str) -> None:

        dump(self.model, model_path)

    def load(self, model_path: str) -> None:

        self.model = load(model_path)

    def evaluate(self, data: np.ndarray, labels: np.ndarray) -> float:

        probabilities = self.model.predict_proba(data)[:, 1]

        probabilities = 1 / (1 + np.exp(-probabilities))

        return float(roc_auc_score(labels, probabilities))

    def predict(self, data: np.ndarray) -> np.ndarray:

        return self.model.predict(data)

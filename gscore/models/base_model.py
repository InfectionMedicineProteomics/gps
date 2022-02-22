import numpy as np
from joblib import dump, load  # type: ignore
from sklearn.metrics import roc_auc_score  # type: ignore


class Scorer:
    def __init__(self):

        pass

    def fit(self, data: np.ndarray, labels: np.ndarray):

        self.model.fit(data, labels)

    def probability(self, data: np.ndarray, input_scores: np.ndarray) -> np.ndarray:

        return self.model.predict_proba(data, input_scores)[:, 1]

    def predict_proba(self, data: np.ndarray) -> np.ndarray:

        return self.model.predict_proba(data)[:, 1]

    def score(self, data: np.ndarray) -> np.ndarray:

        probabilities = self.model.predict_proba(data)[:, 1]

        return np.log(probabilities / (1 - probabilities))

    def save(self, model_path: str):

        dump(self.model, model_path)

    def load(self, model_path: str):

        self.model = load(model_path)

    def evaluate(self, data: np.ndarray, input_scores: np.ndarray, labels: np.ndarray) -> float:

        probabilities = self.probability(data, input_scores)

        return roc_auc_score(labels, probabilities)

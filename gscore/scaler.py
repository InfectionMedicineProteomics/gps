import numpy as np
from sklearn.preprocessing import RobustScaler, MinMaxScaler  # type: ignore

from sklearn.pipeline import Pipeline  # type: ignore

from joblib import dump, load  # type: ignore


class Scaler:

    scaler: Pipeline

    def __init__(self):

        self.scaler = Pipeline(
            [("robust_scaler", RobustScaler()), ("min_max_scaler", MinMaxScaler())]
        )

    def save(self, path: str) -> None:

        dump(self.scaler, path)

    def load(self, path: str) -> None:

        self.scaler = load(path)

    def fit(self, data: np.ndarray) -> None:

        self.scaler.fit(data)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:

        return self.scaler.fit_transform(data)

    def transform(self, data: np.ndarray) -> np.ndarray:

        return self.scaler.transform(data)

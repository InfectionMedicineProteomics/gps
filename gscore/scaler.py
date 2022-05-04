import numpy as np
import numpy.typing as npt

from sklearn.preprocessing import RobustScaler, MinMaxScaler  # type: ignore

from sklearn.pipeline import Pipeline  # type: ignore

from joblib import dump, load  # type: ignore


class Scaler:

    scaler: Pipeline

    def __init__(self) -> None:

        self.scaler = Pipeline(
            [("robust_scaler", RobustScaler()), ("min_max_scaler", MinMaxScaler())]
        )

    def save(self, path: str) -> None:

        dump(self.scaler, path)

    def load(self, path: str) -> None:

        self.scaler = load(path)

    def fit(self, data: npt.NDArray[np.float64]) -> None:

        self.scaler.fit(data)

    def fit_transform(self, data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:

        transformed_data: npt.NDArray[np.float64] = self.scaler.fit_transform(data)

        return transformed_data

    def transform(self, data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:

        transformed_data: npt.NDArray[np.float64] = self.scaler.transform(data)

        return transformed_data

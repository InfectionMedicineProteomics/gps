from __future__ import annotations

import numpy as np

from scipy.interpolate import InterpolatedUnivariateSpline

from sklearn.neighbors import KernelDensity

from typing import TypeVar, Generic, Dict, Tuple

from joblib import dump, load



T = TypeVar("T")

G = TypeVar("G", )


class ScoreDistribution:

    x_axis: np.ndarray
    target_model: KernelDensity
    decoy_model: KernelDensity
    target_function: InterpolatedUnivariateSpline
    decoy_function: InterpolatedUnivariateSpline

    def __init__(self):

        self.target_model = KernelDensity(bandwidth=0.5, kernel="epanechnikov")
        self.decoy_model = KernelDensity(bandwidth=0.5, kernel="epanechnikov")

    def fit(self, data: np.ndarray, labels: np.ndarray):

        self.x_axis = np.linspace(
            start=data.min(),
            stop=data.max(),
            num=1000
        )[:, np.newaxis]

        target_data = data[
            np.argwhere(labels == 1.0)
        ]

        self.target_data = target_data

        decoy_data = data[
            np.argwhere(labels == 0.0)
        ]

        self.decoy_data = decoy_data

        self.target_model.fit(
            target_data
        )

        self.decoy_model.fit(
            decoy_data
        )

        self.target_scores = self.score(model='target')
        self.decoy_scores = self.score(model='decoy')

        self.target_function = InterpolatedUnivariateSpline(
            x=self.x_axis,
            y=self.target_scores,
            ext=2
        )

        self.decoy_function = InterpolatedUnivariateSpline(
            x=self.x_axis,
            y=self.decoy_scores,
            ext=2
        )


    def score(self, model: str):

        if model == "target":

            log_density = self.target_model.score_samples(self.x_axis)

        else:

            log_density = self.decoy_model.score_samples(self.x_axis)

        return np.exp(log_density)


    def calculate_q_vales(self, scores: np.ndarray) -> np.ndarray:

        target_areas = []
        decoy_areas = []

        for score in scores:

            if score > self.decoy_data.max().item():

                decoy_area = 0.0

            else:

                decoy_area = self.decoy_function.integral(
                    a=score,
                    b=self.x_axis[-1].item(),
                )

            if score >= self.target_data.max().item():

                target_area = 1.0

            else:

                target_area = self.target_function.integral(
                    a=score,
                    b=self.x_axis[-1].item()
                )

            target_areas.append(target_area)
            decoy_areas.append(decoy_area)

        target_areas = np.array(target_areas)
        decoy_areas = np.array(decoy_areas)

        total_areas = target_areas + decoy_areas

        q_values = decoy_areas / total_areas

        return q_values


class GlobalDistribution(Generic[T]):

    features: Dict[str, T]
    score_distribution: ScoreDistribution

    def __init__(self) -> None:

        self.features: Dict[str, T] = dict()

    def __contains__(self, item) -> bool:

        return item in self.features

    def __setitem__(self, key: str, value: T) -> None:

        self.features[key] = value

    def compare_score(self, key: str, feature: T) -> None:

        if feature.d_score > self.features[key].d_score:

            self.features[key] = feature

    def _parse_scores(self) -> Tuple[np.ndarray, np.ndarray]:

        self.scores = []
        self.labels = []

        for feature_key, feature in self.features.items():

            self.scores.append(feature.d_score)

            self.labels.append(feature.target)

        self.scores = np.array(self.scores, dtype=np.float64)
        self.labels = np.array(self.labels, dtype=int)

        return self.scores, self.labels

    def fit(self) -> None:

        scores, labels = self._parse_scores()

        self.score_distribution = ScoreDistribution()

        self.score_distribution.fit(
            scores,
            labels
        )

    def save(self, file_path: str) -> None:

        dump(self, file_path)

    @staticmethod
    def load(file_path: str) -> GlobalDistribution:

        return load(file_path)



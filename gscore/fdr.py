from __future__ import annotations

import numpy as np

from scipy.interpolate import InterpolatedUnivariateSpline  # type: ignore

from sklearn.neighbors import KernelDensity  # type: ignore

from typing import TypeVar, Dict, Union

from joblib import dump, load  # type: ignore


from typing import TYPE_CHECKING

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


import numba
from numba import int64, njit, prange, config, threading_layer

import statsmodels.api as sm

if TYPE_CHECKING:
    from gscore.peptides import Peptide
    from gscore.proteins import Protein

T = TypeVar("T")

G = TypeVar(
    "G",
)


class ScoreDistribution:

    x_axis: np.ndarray
    target_model: KernelDensity
    decoy_model: KernelDensity
    target_function: InterpolatedUnivariateSpline
    decoy_function: InterpolatedUnivariateSpline
    scale: bool
    smooth: bool
    transform: Pipeline

    def __init__(self, scale: bool = False, smooth: bool = False):

        self.target_model = KernelDensity(bandwidth=1.0, kernel="epanechnikov")
        self.decoy_model = KernelDensity(bandwidth=1.0, kernel="epanechnikov")
        self.scale = scale
        self.smooth = smooth

    def fit(self, data: np.ndarray, labels: np.ndarray):

        self.x_axis = np.linspace(start=data.min(), stop=data.max(), num=1000)[
            :, np.newaxis
        ]

        if self.scale:

            print("Scaling score distributions.")

            self.transform = Pipeline(
                [
                    # ("standard_scaler", PowerTransformer()),
                    ("robust_scaler", RobustScaler()),
                    # ("robust_scaler", QuantileTransformer(output_distribution="normal")),
                ]
            )

            data = self.transform.fit_transform(data.reshape((-1, 1))).reshape((-1))

            self.x_axis = np.linspace(
                start=data.min() - 1, stop=data.max() + 1, num=1000
            )[:, np.newaxis]

        if self.smooth:

            data = sm.nonparametric.lowess(data, self.x_axis)

        self.target_data = data[np.argwhere(labels == 1.0)]

        self.decoy_data = data[np.argwhere(labels == 0.0)]

        self.target_model.fit(self.target_data)

        self.decoy_model.fit(self.decoy_data)

        self.target_scores = self.score(model="target")
        self.decoy_scores = self.score(model="decoy")

        self.target_function = InterpolatedUnivariateSpline(
            x=self.x_axis, y=self.target_scores, ext=2
        )

        self.decoy_function = InterpolatedUnivariateSpline(
            x=self.x_axis, y=self.decoy_scores, ext=2
        )

    def score(self, model: str):

        if model == "target":

            log_density = self.target_model.score_samples(self.x_axis)

        else:

            log_density = self.decoy_model.score_samples(self.x_axis)

        return np.exp(log_density)

    def calculate_q_values(self, scores: np.ndarray) -> np.ndarray:

        target_areas = []
        decoy_areas = []

        if self.scale:

            print("Scaling scores.")

            scores = self.transform.transform(scores.reshape((-1, 1))).reshape((-1))

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
                    a=score, b=self.x_axis[-1].item()
                )

            target_areas.append(target_area)
            decoy_areas.append(decoy_area)

        target_areas_array = np.array(target_areas)
        decoy_areas_array = np.array(decoy_areas)

        total_areas = target_areas_array + decoy_areas_array

        print(total_areas.min(), total_areas.max())

        q_values = decoy_areas_array / total_areas

        return q_values


class GlobalDistribution:

    features: Dict[str, Union[Peptide, Protein]]
    score_distribution: ScoreDistribution

    def __init__(self) -> None:

        self.features: Dict[str, Union[Peptide, Protein]] = dict()

    def __contains__(self, item) -> bool:

        return item in self.features

    def __setitem__(self, key: str, value: Union[Peptide, Protein]) -> None:

        self.features[key] = value

    def compare_score(self, key: str, feature: Union[Peptide, Protein]) -> None:

        if feature.d_score > self.features[key].d_score:

            self.features[key] = feature

    def _parse_scores(self) -> None:

        scores = []
        labels = []
        feature_keys = []
        probabilities = []

        for feature_key, feature in self.features.items():

            scores.append(feature.d_score)

            labels.append(feature.target)

            feature_keys.append(feature_key)

            probabilities.append(feature.probability)

        self.scores = np.array(scores, dtype=np.float64)
        self.labels = np.array(labels, dtype=int)
        self.feature_keys = np.array(feature_keys, dtype=str)
        self.probabilities = np.array(probabilities, dtype=np.float64)

    def fit(self) -> None:

        self.q_value_map = dict()
        self.score_map = dict()
        self.probability_map = dict()

        self._parse_scores()

        self.score_distribution = ScoreDistribution(scale=False, smooth=False)

        self.score_distribution.fit(self.scores, self.labels)

        self.q_values = self.score_distribution.calculate_q_values(self.scores)

        for idx in range(len(self.q_values)):

            self.q_value_map[self.feature_keys[idx].item()] = self.q_values[idx].item()

            self.score_map[self.feature_keys[idx].item()] = self.scores[idx].item()

            self.probability_map[self.feature_keys[idx].item()] = self.probabilities[
                idx
            ].item()

    def get_score(self, feature_key: str):

        return self.score_map[feature_key]

    def get_q_value(self, feature_key: str):

        return self.q_value_map[feature_key]

    def get_probability(self, feature_key: str):

        return self.probability_map[feature_key]

    def save(self, file_path: str) -> None:

        dump(self, file_path)

    @staticmethod
    def load(file_path: str) -> GlobalDistribution:

        return load(file_path)


@njit(nogil=True)
def _calculate_q_value(labels):

    target_count = 0
    decoy_count = 0
    q_value = 0.0

    for i in range(labels.shape[0]):

        label = labels[i]

        if label == 0:

            target_count += 1

        else:

            decoy_count += 1

    if decoy_count > 0:

        q_value = decoy_count / (decoy_count + target_count)

    return q_value


@njit(parallel=True)
def _calculate_q_values(scores, labels):

    sorted_score_indices = np.argsort(scores)[::-1]

    num_scores = sorted_score_indices.shape[0]

    q_values = np.zeros((num_scores,), dtype=np.float64)

    for i in prange(1, num_scores):

        indices_to_check = np.zeros((i,), dtype=int64)

        for idx in range(i):

            indices_to_check[idx] = sorted_score_indices[idx]

        local_labels = np.zeros((indices_to_check.shape[0],), dtype=int64)

        for idx in range(indices_to_check.shape[0]):

            local_labels[idx] = labels[indices_to_check[idx]]

        q_value = _calculate_q_value(local_labels)

        real_idx = sorted_score_indices[i]

        q_values[real_idx] = q_value

    return q_values


class DecoyCounter:
    def __init__(self, num_threads: int = 10, pi0: float = 1.0):

        numba.set_num_threads(num_threads)

        self.pi0 = pi0

    def calc_q_values(self, scores, labels):

        return _calculate_q_values(scores, labels)

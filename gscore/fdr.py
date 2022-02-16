from __future__ import annotations

import numpy as np

from scipy.interpolate import InterpolatedUnivariateSpline  # type: ignore

from sklearn.neighbors import KernelDensity  # type: ignore

from typing import TypeVar, Dict, Union

from joblib import dump, load  # type: ignore


from typing import TYPE_CHECKING

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

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

        self.target_model = KernelDensity(bandwidth=0.2, kernel="gaussian")
        self.decoy_model = KernelDensity(bandwidth=0.2, kernel="gaussian")
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
                        ("standard_scaler", StandardScaler()),
                        #("min_max_scaler", MinMaxScaler(feature_range=(-10, 10)))
                    ]
                )

            data = self.transform.fit_transform(data.reshape((-1, 1))).reshape((-1))

            self.x_axis = np.linspace(start=data.min() - 1, stop=data.max() + 1, num=1000)[
                          :, np.newaxis
                          ]

        if self.smooth:

            data = sm.nonparametric.lowess(
                data,
                self.x_axis
            )


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

            scores = self.transform.transform(
                scores.reshape((-1, 1))
            ).reshape((-1))

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

        for feature_key, feature in self.features.items():

            scores.append(feature.d_score)

            labels.append(feature.target)

            feature_keys.append(feature_key)

        self.scores = np.array(scores, dtype=np.float64)
        self.labels = np.array(labels, dtype=int)
        self.feature_keys = np.array(feature_keys, dtype=str)

    def fit(self) -> None:

        self.q_value_map = dict()

        self._parse_scores()

        self.score_distribution = ScoreDistribution(
            scale=True,
            smooth=False
        )

        self.score_distribution.fit(self.scores, self.labels)

        self.q_values = self.score_distribution.calculate_q_values(self.scores)

        for idx in range(len(self.q_values)):

            self.q_value_map[self.feature_keys[idx].item()] = self.q_values[idx].item()

    def get_q_value(self, feature_key: str):

        return self.q_value_map[feature_key]

    def save(self, file_path: str) -> None:

        dump(self, file_path)

    @staticmethod
    def load(file_path: str) -> GlobalDistribution:

        return load(file_path)

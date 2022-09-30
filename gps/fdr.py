from __future__ import annotations

import numpy as np


from typing import Dict, Union, Tuple, Any

from joblib import dump, load

from typing import TYPE_CHECKING

import numba

if TYPE_CHECKING:
    from gps.peptides import Peptide
    from gps.proteins import Protein


def _fast_distribution_q_value(target_values, decoy_values, pit):  # type: ignore

    target_area = np.trapz(target_values)

    decoy_area = np.trapz(decoy_values)

    if decoy_area == 0.0 and target_area == 0.0:

        q_value = 0.0

    else:

        decoy_area = decoy_area * pit

        q_value = decoy_area / (decoy_area + target_area)

    return q_value


def _fast_distribution_q_values(scores, target_function, decoy_function, pit):  # type: ignore

    q_values = np.ones((len(scores),), dtype=np.float64)

    max_score = np.max(scores)

    for i in range(len(scores)):

        integral_bounds = np.arange(scores[i], max_score, 0.1)

        target_data = np.interp(integral_bounds, target_function[0], target_function[1])

        decoy_data = np.interp(integral_bounds, decoy_function[0], decoy_function[1])

        q_value = _fast_distribution_q_value(target_data, decoy_data, pit)

        q_values[i] = q_value

    return q_values


class ScoreDistribution:

    pit: float
    X: np.ndarray
    y: np.ndarray
    target_spline: Tuple[np.ndarray, np.ndarray]
    decoy_spline: Tuple[np.ndarray, np.ndarray]
    target_scores: np.ndarray
    decoy_scores: np.ndarray

    def __init__(self, pit: float = 1.0, num_threads: int = 10):

        numba.set_num_threads(num_threads)

        self.pit = pit

    def fit(self, X: np.ndarray, y: np.ndarray) -> ScoreDistribution:

        self.X = X
        self.y = y

        self._estimate_bin_number()

        target_indices = np.argwhere(y == 1)
        decoy_indices = np.argwhere(y == 0)

        self.target_scores = X[target_indices]
        self.decoy_scores = X[decoy_indices]

        self.target_spline = self._fit_function(self.target_scores)
        self.decoy_spline = self._fit_function(self.decoy_scores)

        return self

    def min(self) -> float:

        return float(np.min(self.X))

    def max(self) -> float:

        return float(np.max(self.X))

    def _estimate_bin_number(self) -> None:

        hist, bins = np.histogram(self.X, bins="auto")

        self.num_bins = (bins[1:] + bins[:-1]) / 2

    def _fit_function(self, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        hist, bins = np.histogram(scores, bins=self.num_bins)

        bin_centers = (bins[1:] + bins[:-1]) / 2

        return bin_centers, hist

    def calculate_q_values(self, X: np.ndarray) -> np.ndarray:

        return _fast_distribution_q_values(
            X, self.target_spline, self.decoy_spline, self.pit
        )


class GlobalDistribution:

    features: Dict[str, Union[Peptide, Protein]]
    q_value_map: Dict[str, float]
    score_map: Dict[str, float]
    probability_map: Dict[str, float]
    score_distribution: Union[ScoreDistribution, DecoyCounter]
    pit: float
    count_decoys: bool
    num_threads: int

    def __init__(
        self, pit: float = 1.0, count_decoys: bool = False, num_threads: int = 1
    ) -> None:

        self.features: Dict[str, Union[Peptide, Protein]] = dict()
        self.pit = pit
        self.count_decoys = count_decoys
        self.num_threads = num_threads

    def __contains__(self, item: str) -> bool:

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

        if self.count_decoys:

            self.score_distribution = DecoyCounter(
                num_threads=self.num_threads, pit=self.pit
            )

            self.q_values = self.score_distribution.calc_q_values(
                self.scores, self.labels
            )

        else:

            self.score_distribution = ScoreDistribution(pit=self.pit)

            self.score_distribution.fit(self.scores, self.labels)

            self.q_values = self.score_distribution.calculate_q_values(self.scores)

        for idx in range(len(self.q_values)):

            self.q_value_map[self.feature_keys[idx].item()] = self.q_values[idx].item()

            self.score_map[self.feature_keys[idx].item()] = self.scores[idx].item()

            self.probability_map[self.feature_keys[idx].item()] = self.probabilities[
                idx
            ].item()

    def estimate_pit(self, initial_cutoff: float = 0.01) -> float:

        features = list(self.features.values())

        scores = np.zeros((len(self.features.values()),), dtype=np.float64)

        labels = np.zeros((len(self.features.values()),), dtype=int)

        for i in range(len(features)):
            scores[i] = features[i].d_score
            labels[i] = features[i].target

        score_distribution: Union[DecoyCounter, ScoreDistribution]

        if self.count_decoys:

            score_distribution = DecoyCounter(
                num_threads=self.num_threads, pit=self.pit
            )

            q_values = score_distribution.calc_q_values(scores, labels)

        else:

            score_distribution = ScoreDistribution()

            score_distribution.fit(X=scores, y=labels)

            q_values = score_distribution.calculate_q_values(scores)

        initial_indices = np.argwhere(q_values >= initial_cutoff)

        passed_labels = labels[initial_indices]

        false_target_counts = passed_labels[passed_labels == 1].shape[0]

        decoy_counts = labels[labels == 0].shape[0]

        self.pit = false_target_counts / decoy_counts

        return self.pit

    def get_score(self, feature_key: str) -> float:

        return self.score_map[feature_key]

    def get_q_value(self, feature_key: str) -> float:

        return self.q_value_map[feature_key]

    def get_probability(self, feature_key: str) -> float:

        return self.probability_map[feature_key]

    def save(self, file_path: str) -> None:

        dump(self, file_path)

    @staticmethod
    def load(file_path: str) -> Any:

        return load(file_path)


class DecoyCounter:

    pit: float

    def __init__(self, num_threads: int = 10, pit: float = 1.0) -> None:

        numba.set_num_threads(num_threads)

        self.pit = pit

    def calc_q_values(self, scores: np.ndarray, labels: np.ndarray) -> np.ndarray:

        sorted_score_indices = np.argsort(scores)[::-1]

        num_scores = sorted_score_indices.shape[0]

        q_values = np.zeros((num_scores,), dtype=np.float64)

        num_targets = 0
        num_decoys = 0

        for idx in sorted_score_indices:

            label = labels[idx]

            if label == 1:

                num_targets += 1

            else:

                num_decoys += 1

            decoy_count = num_decoys * self.pit

            q_value = decoy_count / (decoy_count + num_targets)

            q_values[idx] = q_value

        return q_values

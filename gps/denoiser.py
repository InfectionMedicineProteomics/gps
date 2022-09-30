import numpy as np
import numpy.typing as npt

from typing import Optional

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import BaggingClassifier

from gps.preprocess import *


class BaggedDenoiser(BaggingClassifier):  # type: ignore
    def __init__(
        self,
        base_estimator: Optional[Any] = None,
        n_estimators: int = 250,
        max_samples: int = 3,
        n_jobs: int = 5,
        random_state: int = 0,
        class_weights: np.ndarray = np.array([1.0, 1.0]),
    ):

        if not base_estimator:

            base_estimator = SGDClassifier(
                alpha=1e-05,
                average=True,
                loss="log_loss",
                max_iter=500,
                penalty="l2",
                shuffle=True,
                tol=0.0001,
                learning_rate="adaptive",
                eta0=0.001,
                fit_intercept=True,
                random_state=random_state,
                class_weight=dict(enumerate(class_weights)),
            )

        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            bootstrap=True,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def vote(self, noisy_data: np.ndarray, threshold: float = 0.5) -> np.ndarray:

        estimator_probabilities = list()

        for estimator in self.estimators_:

            probabilities = np.where(
                estimator.predict_proba(noisy_data)[:, 1] >= threshold, 1, 0
            )

            estimator_probabilities.append(probabilities)

        estimator_probability_array = np.array(
            estimator_probabilities, dtype=np.float64
        )

        vote_percentages = estimator_probability_array.sum(axis=0) / len(
            self.estimators_
        )

        return vote_percentages

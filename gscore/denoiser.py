import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    RobustScaler,
    MinMaxScaler
)
from sklearn.utils import class_weight


from gscore.utils.ml import *
from gscore.utils import ml
from gscore.scaler import Scaler

class BaggedDenoiser(BaggingClassifier):

    def __init__(
            self,
            base_estimator=None,
            n_estimators=250,
            max_samples=3,
            n_jobs=5,
            random_state=0,
            class_weights: np.ndarray = None
    ):

        if not base_estimator:

            base_estimator = SGDClassifier(
                alpha=1e-05,
                average=True,
                loss='log',
                max_iter=500,
                penalty='l2',
                shuffle=True,
                tol=0.0001,
                learning_rate='adaptive',
                eta0=0.001,
                fit_intercept=True,
                random_state=random_state,
                class_weight=dict(enumerate(class_weights))
            )

        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            bootstrap=True,
            n_jobs=n_jobs,
            random_state=random_state
        )

    def vote(self, noisy_data, threshold=0.5):

        estimator_probabilities = list()

        for estimator in self.estimators_:

            probabilities = np.where(
                estimator.predict_proba(noisy_data)[:,1] >= threshold,
                1,
                0
            )

            estimator_probabilities.append(probabilities)

        estimator_probabilities = np.array(estimator_probabilities)

        vote_percentages = (
            estimator_probabilities.sum(axis=0) /
            len(self.estimators_)
        )

        return vote_percentages

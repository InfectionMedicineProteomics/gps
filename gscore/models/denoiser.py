import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    RobustScaler,
    MinMaxScaler
)

from gscore.datastructures import preprocess_training_data


class BaggedDenoiser(BaggingClassifier):

    def __init__(
            self,
            base_estimator=None,
            n_estimators=250,
            max_samples=3,
            n_jobs=5,
            random_state=0
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
                random_state=random_state
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


def get_denoizer(graph, training_peptides, n_estimators=10, n_jobs=1):

    train_data, train_labels, train_indices = preprocess_training_data(graph, training_peptides)

    scaler = Pipeline([
        ('standard_scaler', RobustScaler()),
        ('min_max_scaler', MinMaxScaler())
    ])

    train_data = scaler.fit_transform(
        train_data
    )

    n_samples = int(len(train_data) * 1.0)  # Change this later based on sample size

    denoizer = BaggedDenoiser(
        max_samples=n_samples,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        random_state=42
    )

    denoizer.fit(
        train_data,
        train_labels.ravel()
    )

    return denoizer, scaler
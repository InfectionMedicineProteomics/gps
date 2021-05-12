import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import BaggingClassifier


class BaggedDenoiser(BaggingClassifier):

    def __init__(
            self,
            base_estimator=None,
            n_estimators=250,
            max_samples=3,
            threads=5,
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
            n_jobs=threads,
            random_state=random_state
        )

    def vote(self, noisy_data, threshold=0.9):

        voted_data = noisy_data.copy()

        estimator_columns = list()

        for estimator_idx, estimator in enumerate(
                self.estimators_
        ):

            estimator_column = f"estimator_{estimator_idx}"

            estimator_columns.append(estimator_column)


            voted_data[estimator_column] = np.where(
                estimator.predict_proba(noisy_data)[:,1] >= threshold,
                1,
                0
            )

        voted_data["vote_percentage"] = voted_data[
            estimator_columns
        ].sum(axis=1) / self.n_estimators

        return voted_data["vote_percentage"]



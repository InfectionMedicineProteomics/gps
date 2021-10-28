import numpy as np
from numba import typed

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    RobustScaler,
    MinMaxScaler
)


from gscore.utils.ml import *


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


class Scaler(Pipeline):


    def __init__(self):

        super(Scaler, self).__init__(
            [
                ('standard_scaler', RobustScaler()),
                ('min_max_scaler', MinMaxScaler())
            ]
        )

def denoise(graph, num_folds, num_classifiers, num_threads, vote_threshold):

    peptide_folds = get_peptide_id_folds(graph, num_folds)

    for fold_num, peptide_fold in enumerate(peptide_folds):

        print(f"Processing fold {fold_num + 1}")

        training_peptides = get_training_peptides(
            peptide_folds=peptide_folds,
            fold_num=fold_num
        )

        scaler = Scaler()

        train_data, train_labels, _ = preprocess_data(
            graph,
            training_peptides,
            use_decoys=False
        )

        train_data = scaler.fit_transform(train_data)

        n_samples = int(len(train_data) * 1.0)

        denoizer = BaggedDenoiser(
            max_samples=n_samples,
            n_estimators=num_classifiers,
            n_jobs=num_threads,
            random_state=42
        )

        denoizer.fit(
            train_data,
            train_labels.ravel()
        )

        testing_scores, testing_labels, testing_keys = preprocess_data(
            graph,
            list(peptide_fold),
            return_all=True

        )

        testing_scores = scaler.transform(
            testing_scores
        )

        class_index = np.where(
            denoizer.classes_ == 1.0
        )[0][0]

        print("Scoring data")

        vote_percentages = denoizer.vote(
            testing_scores,
            threshold=vote_threshold
        )

        probabilities = denoizer.predict_proba(
            testing_scores
        )[:, class_index]

        print("Updating peakgroups")

        graph.update_node_scores(
            typed.List(testing_keys),
            vote_percentages,
            "vote_percentage"
        )

        graph.update_node_scores(
            typed.List(testing_keys),
            probabilities,
            "probability"
        )

        val_scores, val_labels, _ = preprocess_data(
            graph,
            list(peptide_fold),
            use_decoys=False
        )

        val_scores = scaler.transform(
            val_scores
        )

        fold_precision = precision_score(
            y_pred=denoizer.predict(val_scores),
            y_true=val_labels.ravel()
        )

        fold_recall = recall_score(
            y_pred=denoizer.predict(val_scores),
            y_true=val_labels.ravel()
        )

        print(
            f"Fold {fold_num + 1}: Precision = {fold_precision}, Recall = {fold_recall}"
        )


def get_denoizer(graph, training_peptides, n_estimators=10, n_jobs=1):

    train_data, train_labels, _ = preprocess_training_data(graph, training_peptides)

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
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
from gscore import peakgroups
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

def denoise(graph: nx.Graph, num_folds: int, num_classifiers: int, num_threads: int, vote_threshold: float) -> nx.Graph:

    precursor_folds = ml.get_precursor_id_folds(graph, num_folds)

    print(len(precursor_folds))

    total_recall = []
    total_precision = []

    for fold_num, precursor_fold_ids in enumerate(precursor_folds):

        print(f"Processing fold {fold_num + 1}")

        training_precursors = ml.get_training_data(
            folds=precursor_folds,
            fold_num=fold_num
        )

        training_data_targets = peakgroups.get_peakgroups_by_list(
            graph=graph,
            node_list=training_precursors,
            rank=1,
            score_key='var_xcorr_shape_weighted',
            reverse=True
        )

        peakgroup_scores, peakgroup_labels, _ = ml.reformat_data(
            peakgroups=training_data_targets
        )

        train_data, train_labels = shuffle(
            peakgroup_scores,
            peakgroup_labels,
            random_state=42
        )

        scaler = Scaler()

        train_data = scaler.fit_transform(train_data)

        n_samples = int(len(train_data) * 1.0)

        class_weights = class_weight.compute_class_weight(
            class_weight="balanced",
            classes=np.unique(train_labels),
            y=train_labels.ravel()
        )

        denoizer = BaggedDenoiser(
            max_samples=n_samples,
            n_estimators=num_classifiers,
            n_jobs=num_threads,
            random_state=42,
            class_weights=class_weights
        )

        denoizer.fit(
            train_data,
            train_labels.ravel()
        )

        peakgroups_to_score = peakgroups.get_peakgroups_by_list(
            graph=graph,
            node_list=precursor_fold_ids,
            return_all=True
        )

        testing_scores, testing_labels, testing_keys = ml.reformat_data(
            peakgroups=peakgroups_to_score
        )

        testing_scores = scaler.transform(
            testing_scores
        )

        class_index = np.where(
            denoizer.classes_ == 1.0
        )[0][0]

        vote_percentages = denoizer.vote(
            testing_scores,
            threshold=vote_threshold
        )

        probabilities = denoizer.predict_proba(
            testing_scores
        )[:, class_index]

        print("Updating peakgroups", len(probabilities), len(peakgroups_to_score))

        for idx, peakgroup in enumerate(peakgroups_to_score):

            peakgroup.scores['probability'] = probabilities[idx]

            peakgroup.scores['vote_percentage'] = vote_percentages[idx]

        validation_data = peakgroups.get_peakgroups_by_list(
            graph=graph,
            node_list=precursor_fold_ids,
            rank=1,
            score_key='var_xcorr_shape_weighted',
            reverse=True
        )

        val_scores, val_labels, _ = ml.reformat_data(
            peakgroups=validation_data
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

        total_recall.append(fold_recall)
        total_precision.append(fold_precision)

        print(
            f"Fold {fold_num + 1}: Precision = {fold_precision}, Recall = {fold_recall}"
        )

    print(f"Mean recall: {np.mean(total_recall)}, Mean precision: {np.mean(total_precision)}")

    return graph


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
import numpy as np
import pandas as pd

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    MinMaxScaler
)
from sklearn.pipeline import Pipeline
from sklearn.utils import resample

from sklearn.ensemble import BaggingClassifier


class DenoizingFold:
    
    def __init__(self, data_indices, classifiers, scaling_pipeline):

        self.indices = data_indices

        self.classifiers = classifiers

        self.scaling_pipeline = scaling_pipeline


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

    def vote(self, noisy_data):

        voted_data = noisy_data.copy()

        estimator_columns = list()

        for estimator_idx, estimator in enumerate(
                self.estimators_
        ):

            estimator_column = f"estimator_{estimator_idx}"

            estimator_columns.append(estimator_column)

            voted_data[estimator_column] = estimator.predict(noisy_data)

        voted_data["vote_percentage"] = voted_data[
            estimator_columns
        ].sum(axis=1) / self.n_estimators

        return voted_data["vote_percentage"]


# class DenoizingClassifier:
#
#     def __init__(self, target_label, columns):
#
#         self.columns = columns
#
#         self.label = target_label
#
#
#     def fit(self, data, num_classifiers=250, folds=10, match_index='transition_group_id'):
#
#         self.denoizing_folds = list()
#
#         shuffled_peak_groups = data.sample(frac=1)
#
#         split_data = np.array_split(shuffled_peak_groups, folds)
#
#         for idx, fold_data in enumerate(split_data):
#
#             training_data = pd.concat(
#                 [df for i, df in enumerate(split_data) if i != idx]
#             )
#
#             full_pipeline = Pipeline([
#                 ('standard_scaler', RobustScaler()),
#                 ('min_max_scaler', MinMaxScaler())
#             ])
#
#             swath_training_prepared = training_data.copy()
#
#             swath_training_prepared[self.columns] = full_pipeline.fit_transform(
#                 swath_training_prepared[self.columns]
#             )
#
#             n_samples = int(len(swath_training_prepared) * 0.10)
#
#             print(
#                 f"Training for fold {idx} training size={len(swath_training_prepared)} "
#                 f"testing_size={len(fold_data)} {n_samples} for bagging"
#             )
#
#             classifiers = list()
#
#             for bagged_classifier in range(num_classifiers):
#
#                 denoizing_classifier = SGDClassifier(
#                     alpha=1e-05,
#                     average=True,
#                     loss='log',
#                     max_iter=500,
#                     penalty='l2',
#                     shuffle=True,
#                     tol=0.0001,
#                     learning_rate='adaptive',
#                     eta0=0.001,
#                     fit_intercept=True,
#                     random_state=bagged_classifier
#                 )
#
#                 sampled_training_data = resample(
#                     swath_training_prepared,
#                     replace=True,
#                     n_samples=n_samples,
#                     random_state=bagged_classifier
#                 )
#
#                 sgd_classifier = denoizing_classifier.fit(
#                     sampled_training_data[self.columns],
#                     sampled_training_data[self.label]
#                 )
#
#                 classifiers.append(sgd_classifier)
#
#             self.denoizing_folds.append(
#                 DenoizingFold(
#                     data_indices=fold_data,
#                     classifiers=classifiers,
#                     scaling_pipeline=full_pipeline
#                 )
#             )
#
#     # TODO: Need to speed this up
#     def vote(self, data, match_index):
#
#         scored_data = list()
#
#         for denoizing_fold in self.denoizing_folds:
#
#             classifier_columns = list()
#
#             subset = data.loc[
#                 data[match_index].isin(denoizing_fold.indices[match_index])
#             ]
#
#             testing_prepared = subset.copy()
#
#             testing_prepared[self.columns] = denoizing_fold.scaling_pipeline.transform(
#                 testing_prepared[self.columns]
#             )
#
#             for num_classifier, classifier in enumerate(denoizing_fold.classifiers):
#
#
#                 classifier_key = f"num_{num_classifier}"
#
#                 classifier_columns.append(classifier_key)
#
#                 testing_prepared[classifier_key] = classifier.predict(
#                     testing_prepared[self.columns]
#                 )
#
#             testing_prepared['target_vote_percentage'] = testing_prepared[classifier_columns].sum(axis=1) / len(classifier_columns)
#
#             scored_data.append(
#                 testing_prepared[
#                     ['transition_group_id', 'target_vote_percentage']
#                 ]
#             )
#
#         scored_data = pd.concat(
#             scored_data,
#             ignore_index=True
#         )
#
#         return scored_data['target_vote_percentage']
#
#
#     def predict_average_score(self, data, match_index):
#
#         scored_data = list()
#
#         for denoizing_fold in self.denoizing_folds:
#
#             classifier_columns = list()
#
#             subset = data.loc[
#                 data[match_index].isin(denoizing_fold.indices[match_index])
#             ]
#
#             testing_prepared = subset.copy()
#
#             testing_prepared[self.columns] = denoizing_fold.scaling_pipeline.transform(
#                 testing_prepared[self.columns]
#             )
#
#             for num_classifier, classifier in enumerate(denoizing_fold.classifiers):
#                 classifier_key = f"num_{num_classifier}"
#
#                 classifier_columns.append(classifier_key)
#
#                 testing_prepared[classifier_key] = classifier.decision_function(
#                     testing_prepared[self.columns]
#                 )
#
#             testing_prepared['average_combined_score'] = testing_prepared[classifier_columns].mean(axis=1)
#
#
#             scored_data.append(
#                 testing_prepared[
#                     ['transition_group_id', 'average_combined_score']
#                 ]
#             )
#
#         scored_data = pd.concat(
#             scored_data,
#             ignore_index=True
#         )
#
#         return scored_data['average_combined_score']

from functools import partial
from typing import Tuple

from abc import ABC, abstractmethod

from joblib import dump, load

import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.preprocessing import (
    RobustScaler,
    MinMaxScaler
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score


from imblearn.ensemble import BalancedBaggingClassifier, EasyEnsembleClassifier

from gscore.scaler import Scaler

from xgboost import XGBClassifier

from gscore.utils import ml

MODELS = {
    "adaboost": AdaBoostClassifier
}

class Scorer:

    def fit(self, data: np.ndarray, labels: np.ndarray):

        self.model.fit(
            data,
            labels
        )

    def probability(self, data: np.ndarray) -> np.ndarray:

        return self.model.predict_proba(data)[:, 1]

    def score(self, data: np.ndarray) -> np.ndarray:

        probabilities = self.model.predict_proba(data)[:, 1]

        return np.log(probabilities / (1 - probabilities))

    def save(self, model_path: str):

        dump(self.model, model_path)

    def load(self, model_path: str):

        self.model = load(model_path)


class SGDScorer(Scorer):

    model: SGDClassifier

    def __init__(self, class_weights: np.ndarray):

        self.model = SGDClassifier(
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
            random_state=42,
            class_weight=dict(enumerate(class_weights))
        )


class XGBoostScorer(Scorer):

    model: XGBClassifier

    def __init__(self, scale_pos_weight: float):

        self.model = XGBClassifier(
            n_estimators=100,
            verbosity=1,
            objective="binary:logistic",
            n_jobs=10,
            random_state=42,
            scale_pos_weight=scale_pos_weight
        )


class EasyEnsembleScorer(Scorer):

    model: EasyEnsembleClassifier
    submodel: SGDClassifier

    def __init__(self, class_weights: np.ndarray):

        self.submodel = SGDClassifier(
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
            random_state=42,
            class_weight=dict(enumerate(class_weights))
        )

        self.model = EasyEnsembleClassifier(
            base_estimator=self.submodel,
            n_estimators=100,
            sampling_strategy='auto',
            random_state=42,
            n_jobs=10,
            verbose=True
        )




class BalancedBaggingScorer(Scorer):

    model: BalancedBaggingClassifier
    submodel: SGDClassifier

    def __init__(self, class_weights: np.ndarray):

        self.submodel = SGDClassifier(
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
            random_state=42,
            class_weight=dict(enumerate(class_weights))
        )

        self.model = BalancedBaggingClassifier(
            base_estimator=self.submodel,
            n_estimators=100,
            bootstrap=True,
            sampling_strategy='auto',
            random_state=42,
            n_jobs=10,
            verbose=True
        )


class GradientBoostingScorer(Scorer):

    model: GradientBoostingClassifier

    def __init__(self):

        self.model = GradientBoostingClassifier()


class AdaBoostSGDScorer(Scorer):

    model: AdaBoostClassifier
    submodel: SGDClassifier

    def __init__(self, class_weights: np.ndarray):

        self.submodel = SGDClassifier(
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
            random_state=42,
            class_weight=dict(enumerate(class_weights))
        )

        self.model = AdaBoostClassifier(
            base_estimator=self.submodel,
            n_estimators=100,
            learning_rate=1.0,
            algorithm="SAMME.R",
            random_state=42
        )


def train_model(data: np.ndarray, labels: np.ndarray, model: str, scaling_pipeline: Pipeline) -> None:

    data = scaling_pipeline.fit_transform(data)

    model.fit(data, labels.ravel())


def evaluate_model(data: np.ndarray, labels: np.ndarray, model: Scorer, scaling_pipeline: Pipeline) -> float:

    data = scaling_pipeline.transform(data)

    probabilities = model.probability(data)

    return roc_auc_score(labels, probabilities)


def score_run(precursors, model_path: str, scaler_path: str):

    scoring_model = Scorer()

    scoring_model.load(model_path)

    pipeline = Scaler()

    pipeline.load(scaler_path)

    all_peakgroups = precursors.get_all_peakgroups()

    all_data_scores, all_data_labels, all_data_indices = ml.reformat_data(
        all_peakgroups,
        include_score_columns=True
    )

    all_data_scores = pipeline.transform(all_data_scores)

    model_scores = score(all_data_scores, scoring_model, pipeline)

    for idx, peakgroup in enumerate(all_peakgroups):

        peakgroup.scores['d_score'] = model_scores[idx]

    return precursors


if __name__ == '__main__':
    import glob

    from gscore.parsers import osw
    from gscore.parsers import queries
    from gscore import peakgroups
    from sklearn.utils import shuffle

    from gscore.utils import ml
    from gscore.denoiser import denoise

    from gscore.scaler import Scaler

    from gscore.utils.connection import Connection

    from gscore.parsers.queries import (
        CreateIndex,
        SelectPeakGroups
    )

    from gscore.distributions import ScoreDistribution

    all_sample_data = []

    osw_files = glob.glob("/home/aaron/projects/ghost/data/spike_in/openswath/*.osw")

    for osw_file in osw_files[:1]:

        print(f"Processing {osw_file}")

        with osw.OSWFile(osw_file) as conn:
            precursors = conn.fetch_subscore_records(query=queries.SelectPeakGroups.FETCH_ALL_DENOIZED_DATA)

        print("Scoring")

        precursors = score_run(
            precursors=precursors,
            model_path="/home/aaron/projects/gscorer/notebooks/xgb_test.model",
            scaler_path="/home/aaron/projects/gscorer/notebooks/scaler_pipeline.pkl"
        )

        target_peakgroups = precursors.get_target_peakgroups_by_rank(
            rank=1,
            score_key="d_score",
            reverse=True
        )

        decoy_peakgroups = precursors.get_decoy_peakgroups(
            sort_key="d_score"
        )

        modelling_peakgroups = target_peakgroups + decoy_peakgroups

        all_data_scores, all_data_labels = ml.reformat_distribution_data(
            modelling_peakgroups,
            score_column="d_score"
        )

        score_distribution = ScoreDistribution()

        score_distribution.fit(
            all_data_scores,
            all_data_labels
        )

        print("here")

        q_values = score_distribution.calculate_q_vales(np.array([6.0]))

        print(q_values)


# from tensorflow import keras
#
#
# ADAM_OPTIMIZER = keras.optimizers.Adam(learning_rate=0.001)
#
# EARLY_STOPPING_CB = keras.callbacks.EarlyStopping(
#     patience=5,
#     restore_best_weights=True
# )
#
#
# class TargetScoringModel(keras.Model):
#
#     RegularizedDense = partial(
#         keras.layers.Dense,
#         activation='elu',
#         kernel_initializer='he_normal',
#         kernel_regularizer=keras.regularizers.l2(),
#         dtype='float64'
#     )
#
#     def __init__(self, input_dim, **kwargs):
#         super().__init__(**kwargs)
#         self.dense_1 = self.RegularizedDense(30, input_shape=input_dim)
#         self.dense_2 = self.RegularizedDense(30)
#         self.dense_3 = self.RegularizedDense(30)
#         self.dense_4 = self.RegularizedDense(30)
#         self.score_output = keras.layers.Dense(1, activation='sigmoid', dtype='float64')
#
#     def call(self, inputs):
#         x = self.dense_1(inputs)
#         x = self.dense_2(x)
#         x = self.dense_3(x)
#         x = self.dense_4(x)
#         return self.score_output(x)

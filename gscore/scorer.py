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


from xgboost import XGBClassifier

MODELS = {
    "adaboost": AdaBoostClassifier
}

class Scorer(ABC):

    @abstractmethod
    def fit(self, data: np.ndarray) -> None:

        raise NotImplementedError

    @abstractmethod
    def probability(self, data: np.ndarray) -> np.ndarray:

        raise NotImplementedError

    @abstractmethod
    def score(self) -> np.ndarray:

        raise NotImplementedError

    @abstractmethod
    def save(self):

        raise NotImplementedError

    @abstractmethod
    def load(self):

        raise NotImplementedError


class SGDScorer(Scorer):

    model: SGDClassifier

    def __init__(self, class_weights: ndarray):

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



class GradientBoostingScorer(Scorer):

    model: GradientBoostingClassifier

    def __init__(self):

        self.model = GradientBoostingClassifier()

    def fit(self, data: np.ndarray, labels: np.ndarray):

        self.model.fit(
            data,
            labels
        )

    def probability(self, data: np.ndarray) -> np.ndarray:

        return self.model.predict_proba(data)[:, 1]

    def score(self, data: np.ndarray) -> np.ndarray:

        probabilities = self.model.predict_proba(data)[:,1]

        return np.log(probabilities / (1 - probabilities))

    def save(self, model_path: str):

        dump(self.model, model_path)

    def load(self, model_path: str):

        self.model = load(model_path)


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

    def fit(self, data: np.ndarray, labels: np.ndarray):

        self.model.fit(
            data,
            labels
        )

    def probability(self, data: np.ndarray) -> np.ndarray:

        return self.model.predict_proba(data)[:, 1]

    def score(self, data: np.ndarray) -> np.ndarray:

        probabilities = self.model.predict_proba(data)[:,1]

        return np.log(probabilities / (1 - probabilities))

    def save(self, model_path: str):

        dump(self.model, model_path)

    def load(self, model_path: str):

        self.model = load(model_path)


def train_model(data: np.ndarray, labels: np.ndarray, model: str, scaling_pipeline: Pipeline) -> None:

    data = scaling_pipeline.fit_transform(data)

    model.fit(data, labels.ravel())


def evaluate_model(data: np.ndarray, labels: np.ndarray, model: Scorer, scaling_pipeline: Pipeline) -> float:

    data = scaling_pipeline.transform(data)

    probabilities = model.probability(data)

    return roc_auc_score(labels, probabilities)


def score(data: np.ndarray, model: Scorer, scaling_pipeline: Pipeline) -> np.ndarray:

    data = scaling_pipeline.transform(data)

    logit_scores = model.score(data)

    return logit_scores



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

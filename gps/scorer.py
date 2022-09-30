from joblib import dump, load

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from imblearn.ensemble import BalancedBaggingClassifier, EasyEnsembleClassifier

from xgboost import XGBClassifier


import numpy as np


MODELS = {"adaboost": AdaBoostClassifier}


class Scorer:
    def fit(self, data: np.ndarray, labels: np.ndarray) -> None:

        self.model.fit(data, labels)

    def probability(self, data: np.ndarray) -> np.ndarray:

        probabilities = self.model.predict_proba(data)[:, 1]

        probabilities = 1 / (1 + np.exp(-probabilities))

        return probabilities

    def predict_proba(self, data: np.ndarray) -> np.ndarray:

        probabilities = self.model.predict_proba(data)[:, 1]

        probabilities = 1 / (1 + np.exp(-probabilities))

        return probabilities

    def score(self, data: np.ndarray) -> np.ndarray:

        return self.model.predict_proba(data)[:, 1]

    def save(self, model_path: str) -> None:

        dump(self.model, model_path)

    def load(self, model_path: str) -> None:

        self.model = load(model_path)

    def evaluate(self, data: np.ndarray, labels: np.ndarray) -> float:

        probabilities = self.model.predict_proba(data)[:, 1]

        probabilities = 1 / (1 + np.exp(-probabilities))

        return float(roc_auc_score(labels, probabilities))


class SGDScorer(Scorer):

    model: SGDClassifier

    def __init__(self, class_weights: np.ndarray):

        self.model = SGDClassifier(
            alpha=1e-05,
            average=True,
            loss="log",
            max_iter=500,
            penalty="l2",
            shuffle=True,
            tol=0.0001,
            learning_rate="adaptive",
            eta0=0.001,
            fit_intercept=True,
            random_state=42,
            class_weight=dict(enumerate(class_weights)),
        )


class XGBoostScorer(Scorer):

    model: XGBClassifier

    def __init__(self, scale_pos_weight: float):

        self.model = XGBClassifier(
            n_estimators=100,
            verbosity=1,
            objective="binary:logitraw",
            n_jobs=10,
            random_state=42,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
        )


class EasyEnsembleScorer(Scorer):

    model: EasyEnsembleClassifier
    submodel: SGDClassifier

    def __init__(self, class_weights: np.ndarray):

        self.submodel = SGDClassifier(
            alpha=1e-05,
            average=True,
            loss="log",
            max_iter=500,
            penalty="l2",
            shuffle=True,
            tol=0.0001,
            learning_rate="adaptive",
            eta0=0.001,
            fit_intercept=True,
            random_state=42,
            class_weight=dict(enumerate(class_weights)),
        )

        self.model = EasyEnsembleClassifier(
            base_estimator=self.submodel,
            n_estimators=100,
            sampling_strategy="auto",
            random_state=42,
            n_jobs=10,
            verbose=True,
        )


class XGBEnsembleScorer(Scorer):

    model: EasyEnsembleClassifier
    submodel: SGDClassifier

    def __init__(self, scale_pos_weight: float):

        self.submodel = XGBClassifier(
            n_estimators=10,
            verbosity=1,
            objective="binary:logistic",
            n_jobs=5,
            random_state=42,
            scale_pos_weight=scale_pos_weight,
        )

        self.model = EasyEnsembleClassifier(
            base_estimator=self.submodel,
            n_estimators=10,
            sampling_strategy="auto",
            random_state=42,
            n_jobs=2,
            verbose=True,
        )


class BalancedBaggingScorer(Scorer):

    model: BalancedBaggingClassifier
    submodel: SGDClassifier

    def __init__(self, class_weights: np.ndarray):

        self.submodel = SGDClassifier(
            alpha=1e-05,
            average=True,
            loss="log",
            max_iter=500,
            penalty="l2",
            shuffle=True,
            tol=0.0001,
            learning_rate="adaptive",
            eta0=0.001,
            fit_intercept=True,
            random_state=42,
            class_weight=dict(enumerate(class_weights)),
        )

        self.model = BalancedBaggingClassifier(
            base_estimator=self.submodel,
            n_estimators=100,
            bootstrap=True,
            sampling_strategy="auto",
            random_state=42,
            n_jobs=10,
            verbose=True,
        )


class GradientBoostingScorer(Scorer):

    model: GradientBoostingClassifier

    def __init__(self) -> None:

        self.model = GradientBoostingClassifier()


class AdaBoostSGDScorer(Scorer):

    model: AdaBoostClassifier
    submodel: SGDClassifier

    def __init__(self, class_weights: np.ndarray):

        self.submodel = SGDClassifier(
            alpha=1e-05,
            average=True,
            loss="log",
            max_iter=500,
            penalty="l2",
            shuffle=True,
            tol=0.0001,
            learning_rate="adaptive",
            eta0=0.001,
            fit_intercept=True,
            random_state=42,
            class_weight=dict(enumerate(class_weights)),
        )

        self.model = AdaBoostClassifier(
            base_estimator=self.submodel,
            n_estimators=100,
            learning_rate=1.0,
            algorithm="SAMME.R",
            random_state=42,
        )

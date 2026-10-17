"""The three standard models. Deliberately un-tuned defaults: the question is
how much any reasonable model loses when the network changes, not which model
wins a leaderboard."""
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .features import signed_log1p

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None


def build(name: str, y_train: np.ndarray, seed: int):
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, class_weight="balanced_subsample",
            n_jobs=-1, random_state=seed)
    if name == "xgb":
        if XGBClassifier is None:
            raise ImportError("pip install xgboost")
        pos = max(int(y_train.sum()), 1)
        neg = max(int(len(y_train) - y_train.sum()), 1)
        return XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, tree_method="hist", eval_metric="logloss",
            scale_pos_weight=neg / pos, n_jobs=-1, random_state=seed)
    if name == "mlp":
        return make_pipeline(
            FunctionTransformer(signed_log1p), StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(128, 64), early_stopping=True,
                          max_iter=60, batch_size=512, random_state=seed))
    raise ValueError(name)


def fit_and_scorer(name, X, y, seed):
    model = build(name, y, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
    return lambda Z: model.predict_proba(Z)[:, 1]


MODEL_NAMES = {"rf": "Random Forest", "xgb": "XGBoost", "mlp": "MLP"}

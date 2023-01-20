from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import optuna
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass(frozen=True)
class ModelSpec:
    name: str
    build: Callable[[], Any]
    param_grid: dict[str, list]
    param_distributions_random: dict[str, list]
    param_distributions_optuna: dict[str, Any]


def _rf() -> RandomForestClassifier:
    return RandomForestClassifier(random_state=0, n_jobs=-1)


def _gb() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(random_state=0)


def _lr_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(random_state=0, max_iter=2000, solver="lbfgs"),
            ),
        ]
    )


def _svc_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", SVC(random_state=0, probability=True)),
        ]
    )


MODELS: dict[str, ModelSpec] = {
    "random_forest": ModelSpec(
        name="random_forest",
        build=_rf,
        param_grid={
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 8, 16, 24],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        param_distributions_random={
            "n_estimators": [50, 100, 150, 200],
            "max_depth": [None, 6, 12, 18, 24],
            "min_samples_split": [2, 4, 8, 16],
            "min_samples_leaf": [1, 2, 4],
        },
        param_distributions_optuna={
            "n_estimators": optuna.distributions.IntDistribution(50, 250),
            "max_depth": optuna.distributions.IntDistribution(4, 40),
            "min_samples_split": optuna.distributions.IntDistribution(2, 20),
            "min_samples_leaf": optuna.distributions.IntDistribution(1, 8),
        },
    ),
    "gradient_boosting": ModelSpec(
        name="gradient_boosting",
        build=_gb,
        param_grid={
            "n_estimators": [50, 100, 150],
            "learning_rate": [0.05, 0.1, 0.2],
            "max_depth": [2, 3, 5],
            "subsample": [0.8, 1.0],
        },
        param_distributions_random={
            "n_estimators": [50, 100, 150, 200],
            "learning_rate": [0.03, 0.05, 0.1, 0.15, 0.2],
            "max_depth": [2, 3, 4, 5],
            "subsample": [0.7, 0.8, 0.9, 1.0],
        },
        param_distributions_optuna={
            "n_estimators": optuna.distributions.IntDistribution(50, 250),
            "learning_rate": optuna.distributions.FloatDistribution(0.01, 0.3, log=True),
            "max_depth": optuna.distributions.IntDistribution(2, 8),
            "subsample": optuna.distributions.FloatDistribution(0.6, 1.0),
        },
    ),
    "logistic_regression": ModelSpec(
        name="logistic_regression",
        build=_lr_pipeline,
        param_grid={
            "clf__C": [0.01, 0.1, 1.0, 10.0],
            "clf__penalty": ["l2"],
        },
        param_distributions_random={
            "clf__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "clf__penalty": ["l2"],
        },
        param_distributions_optuna={
            "clf__C": optuna.distributions.FloatDistribution(1e-3, 1e2, log=True),
        },
    ),
    "svc_rbf": ModelSpec(
        name="svc_rbf",
        build=_svc_pipeline,
        param_grid={
            "clf__C": [0.1, 1.0, 10.0],
            "clf__gamma": ["scale", "auto", 0.01, 0.1],
        },
        param_distributions_random={
            "clf__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "clf__gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1.0],
        },
        param_distributions_optuna={
            "clf__C": optuna.distributions.FloatDistribution(1e-2, 1e2, log=True),
            "clf__gamma": optuna.distributions.FloatDistribution(1e-4, 1.0, log=True),
        },
    ),
}


def get_model(name: str) -> ModelSpec:
    if name not in MODELS:
        allowed = ", ".join(sorted(MODELS))
        raise ValueError(f"Unknown model {name!r}. Choose one of: {allowed}")
    return MODELS[name]

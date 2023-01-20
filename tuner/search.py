from __future__ import annotations

import json
from typing import Any

import numpy as np
from sklearn.metrics import get_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.utils.multiclass import type_of_target

from tuner.models import ModelSpec, get_model


def _scorer_for_metric(metric: str, y) -> object:
    if metric == "roc_auc":
        tt = type_of_target(y)
        if tt == "binary":
            return get_scorer("roc_auc")
        return get_scorer("roc_auc_ovr_weighted")
    return get_scorer(metric)


def _cv_splits(y, cv: int, random_state: int) -> StratifiedKFold:
    return StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)


def run_grid(
    spec: ModelSpec,
    X,
    y,
    *,
    cv: int,
    metric: str,
    random_state: int,
    n_jobs: int,
) -> dict[str, Any]:
    est = spec.build()
    scorer = _scorer_for_metric(metric, y)
    search = GridSearchCV(
        est,
        spec.param_grid,
        scoring=scorer,
        cv=_cv_splits(y, cv, random_state),
        n_jobs=n_jobs,
        refit=True,
        error_score="raise",
    )
    search.fit(X, y)
    out = _result_dict("grid", spec.name, search, metric, X)
    out["_best_estimator"] = search.best_estimator_
    return out


def run_random(
    spec: ModelSpec,
    X,
    y,
    *,
    cv: int,
    metric: str,
    random_state: int,
    n_iter: int,
    n_jobs: int,
) -> dict[str, Any]:
    est = spec.build()
    scorer = _scorer_for_metric(metric, y)
    search = RandomizedSearchCV(
        est,
        spec.param_distributions_random,
        n_iter=n_iter,
        scoring=scorer,
        cv=_cv_splits(y, cv, random_state),
        n_jobs=n_jobs,
        random_state=random_state,
        refit=True,
        error_score="raise",
    )
    search.fit(X, y)
    out = _result_dict("random", spec.name, search, metric, X)
    out["_best_estimator"] = search.best_estimator_
    return out


def run_optuna(
    spec: ModelSpec,
    X,
    y,
    *,
    cv: int,
    metric: str,
    random_state: int,
    n_trials: int,
    n_jobs: int,
) -> dict[str, Any]:
    try:
        from optuna.integration import OptunaSearchCV
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Optuna sklearn integration is required for --method optuna. "
            "Install with: pip install 'optuna-integration[sklearn]'"
        ) from e
    est = spec.build()
    scorer = _scorer_for_metric(metric, y)
    search = OptunaSearchCV(
        est,
        spec.param_distributions_optuna,
        cv=_cv_splits(y, cv, random_state),
        scoring=scorer,
        n_trials=n_trials,
        random_state=random_state,
        n_jobs=n_jobs,
        refit=True,
        error_score="raise",
    )
    search.fit(X, y)
    out = _result_dict("optuna", spec.name, search, metric, X)
    out["_best_estimator"] = search.best_estimator_
    return out


def _result_dict(method: str, model_name: str, search, metric: str, X) -> dict[str, Any]:
    best_params = search.best_params_
    best_cv = float(search.best_score_)
    out: dict[str, Any] = {
        "method": method,
        "model": model_name,
        "metric": metric,
        "best_cv_score": best_cv,
        "best_params": {k: _jsonable(v) for k, v in best_params.items()},
    }
    est = getattr(search, "best_estimator_", None)
    if est is None:
        return out
    inner = est
    if hasattr(est, "named_steps"):
        inner = est.named_steps.get("clf", list(est.named_steps.values())[-1])
    if hasattr(inner, "feature_importances_"):
        imps = np.asarray(inner.feature_importances_)
        if hasattr(est, "feature_names_in_"):
            names = list(est.feature_names_in_)
        elif hasattr(X, "columns"):
            names = list(X.columns)
        else:
            names = [f"f{i}" for i in range(len(imps))]
        pairs = sorted(zip(names, imps, strict=False), key=lambda t: -t[1])
        out["top_features"] = [{"feature": n, "importance": float(v)} for n, v in pairs[:12]]
    return out


def _jsonable(v: Any) -> Any:
    if v is None or isinstance(v, (str, bool, int)):
        return v
    if isinstance(v, float):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, np.generic):
        return v.item()
    return str(v)


def run(
    model_name: str,
    method: str,
    X,
    y,
    *,
    cv: int,
    metric: str,
    random_state: int,
    n_iter: int,
    n_trials: int,
    n_jobs: int,
) -> dict[str, Any]:
    spec = get_model(model_name)
    if method == "grid":
        return run_grid(spec, X, y, cv=cv, metric=metric, random_state=random_state, n_jobs=n_jobs)
    if method == "random":
        return run_random(
            spec,
            X,
            y,
            cv=cv,
            metric=metric,
            random_state=random_state,
            n_iter=n_iter,
            n_jobs=n_jobs,
        )
    if method == "optuna":
        return run_optuna(
            spec,
            X,
            y,
            cv=cv,
            metric=metric,
            random_state=random_state,
            n_trials=n_trials,
            n_jobs=n_jobs,
        )
    raise ValueError(f"Unknown method {method!r}")


def dump_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

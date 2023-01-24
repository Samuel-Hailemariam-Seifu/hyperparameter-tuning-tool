"""Shared helpers for reporting scores (CLI + UI)."""

from __future__ import annotations

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.utils.multiclass import type_of_target


def public_result(result: dict) -> dict:
    return {k: v for k, v in result.items() if not str(k).startswith("_")}


def holdout_score(est, X, y, metric: str) -> float:
    tt = type_of_target(y)
    if metric == "accuracy":
        return float(accuracy_score(y, est.predict(X)))
    if metric == "f1_macro":
        return float(f1_score(y, est.predict(X), average="macro"))
    if metric == "roc_auc":
        if tt == "binary":
            proba = est.predict_proba(X)
            col = proba[:, 1] if proba.shape[1] == 2 else proba.ravel()
            return float(roc_auc_score(y, col))
        proba = est.predict_proba(X)
        return float(
            roc_auc_score(y, proba, multi_class="ovr", average="weighted"),
        )
    raise ValueError(metric)

from __future__ import annotations

import argparse
import json

from rich.console import Console
from rich.table import Table
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.utils.multiclass import type_of_target

from tuner.data import load_data
from tuner.models import MODELS
from tuner.search import dump_json, run

console = Console()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hpt-tune",
        description="Hyperparameter search for sklearn classifiers (grid, random, Optuna).",
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to CSV (classification target column; default: last column).",
    )
    src.add_argument(
        "--dataset",
        type=str,
        default="breast_cancer",
        choices=sorted(["iris", "wine", "breast_cancer"]),
        help="Built-in sklearn dataset when --data is omitted.",
    )
    p.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target column name for CSV (default: last column).",
    )
    p.add_argument(
        "--model",
        type=str,
        required=True,
        choices=sorted(MODELS.keys()),
        help="Estimator preset.",
    )
    p.add_argument(
        "--method",
        type=str,
        default="random",
        choices=["grid", "random", "optuna"],
        help="Search strategy.",
    )
    p.add_argument(
        "--metric",
        type=str,
        default="accuracy",
        choices=["accuracy", "f1_macro", "roc_auc"],
        help="Scoring metric for CV.",
    )
    p.add_argument("--cv", type=int, default=5, help="Cross-validation folds.")
    p.add_argument(
        "--n-iter",
        type=int,
        default=32,
        help="Iterations for randomized search.",
    )
    p.add_argument(
        "--n-trials",
        type=int,
        default=40,
        help="Trials for Optuna search.",
    )
    p.add_argument(
        "--test-size",
        type=float,
        default=0.0,
        help="Optional holdout fraction for a final score (0 = CV only).",
    )
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs for sklearn search (-1 = all cores).",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write JSON results to this path.",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON to stdout instead of a table.",
    )
    return p


def _holdout_score(est, X, y, metric: str) -> float:
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


def _public_dict(result: dict) -> dict:
    return {k: v for k, v in result.items() if not str(k).startswith("_")}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    X_train, X_hold, y_train, y_hold = load_data(
        csv_path=args.data,
        dataset=args.dataset,
        target=args.target,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    result = run(
        args.model,
        args.method,
        X_train,
        y_train,
        cv=args.cv,
        metric=args.metric,
        random_state=args.random_state,
        n_iter=args.n_iter,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
    )

    if X_hold is not None and y_hold is not None:
        est = result.get("_best_estimator")
        if est is not None:
            result["holdout_score"] = _holdout_score(est, X_hold, y_hold, args.metric)

    if args.output:
        dump_json(args.output, _public_dict(result))

    if args.json:
        print(json.dumps(_public_dict(result), indent=2))
        return 0

    table = Table(title="Hyperparameter search", show_header=True, header_style="bold")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Best CV score", f"{result['best_cv_score']:.6f}")
    table.add_row("Metric", result["metric"])
    table.add_row("Model", result["model"])
    table.add_row("Method", result["method"])
    table.add_row("Best params", json.dumps(result["best_params"], indent=2))
    if "holdout_score" in result:
        table.add_row("Holdout score", f"{result['holdout_score']:.6f}")
    if "top_features" in result:
        top = ", ".join(f"{x['feature']}" for x in result["top_features"][:5])
        table.add_row("Top features", top)
    console.print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

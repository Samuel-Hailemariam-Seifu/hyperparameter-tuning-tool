from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


DATASETS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
}


def prepare_features(X: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categoricals; pass through numeric-only frames."""
    return pd.get_dummies(X, drop_first=False)


def load_builtin(name: str) -> tuple[pd.DataFrame, pd.Series]:
    if name not in DATASETS:
        allowed = ", ".join(sorted(DATASETS))
        raise ValueError(f"Unknown dataset {name!r}. Choose one of: {allowed}")
    bundle = DATASETS[name]()
    X = pd.DataFrame(bundle.data, columns=bundle.feature_names)
    y = pd.Series(bundle.target, name="target")
    return X, y


def load_csv_dataframe(df: pd.DataFrame, target: str | None) -> tuple[pd.DataFrame, pd.Series]:
    if target is None:
        target = str(df.columns[-1])
    if target not in df.columns:
        raise ValueError(f"Target column {target!r} not in CSV columns: {list(df.columns)}")
    y_raw = df[target]
    X = df.drop(columns=[target])
    X = prepare_features(X)
    y = pd.Series(LabelEncoder().fit_transform(y_raw), name=target)
    return X, y


def load_csv(path: str, target: str | None) -> tuple[pd.DataFrame, pd.Series]:
    p = Path(path).expanduser()
    if not p.is_file():
        cwd = Path.cwd()
        raise FileNotFoundError(
            f"CSV not found: {p.resolve()!s}; cwd={cwd!s}; "
            "use a real file path for --data, or omit --data and set "
            "--dataset to iris, wine, or breast_cancer."
        )
    df = pd.read_csv(p)
    return load_csv_dataframe(df, target)


def load_csv_bytes(raw: bytes, target: str | None) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(io.BytesIO(raw))
    return load_csv_dataframe(df, target)


def load_data(
    *,
    csv_path: str | None = None,
    csv_bytes: bytes | None = None,
    dataset: str = "breast_cancer",
    target: str | None = None,
    test_size: float = 0.0,
    random_state: int = 42,
) -> tuple:
    if csv_bytes is not None:
        X, y = load_csv_bytes(csv_bytes, target)
    elif csv_path:
        X, y = load_csv(csv_path, target)
    else:
        X, y = load_builtin(dataset)
    if test_size and test_size > 0:
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
    return X, None, y, None

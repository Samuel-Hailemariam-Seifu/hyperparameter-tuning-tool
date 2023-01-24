"""
Lay-friendly web UI for hyperparameter search (run with: streamlit run tuner/ui_app.py).
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from tuner.data import load_data
from tuner.evaluate import holdout_score, public_result
from tuner.models import MODELS
from tuner.search import run

DATASET_HELP = {
    "breast_cancer": "Demo: predict tumor benign vs malignant from measurements (realistic example).",
    "iris": "Demo: classify flower species from petal/sepal sizes (tiny dataset, runs fast).",
    "wine": "Demo: classify wine type from chemical measurements.",
}

MODEL_HELP = {
    "random_forest": "Builds many decision trees and averages their votes — robust and popular.",
    "gradient_boosting": "Trees are added one after another to fix mistakes — often very accurate.",
    "logistic_regression": "A simple, fast linear model — good baseline.",
    "svc_rbf": "Can draw smooth boundaries between classes — strong when you have enough data.",
}

METHOD_HELP = {
    "random": "Tries random combinations of settings. **Good default** — fast enough and works well.",
    "grid": "Tries **every** combination in a fixed list. Thorough but can take a long time.",
    "optuna": "Uses a smarter search that learns from past tries. Can find better settings, takes longer.",
}

METRIC_HELP = {
    "accuracy": "What fraction of rows the model classifies correctly.",
    "f1_macro": "Balances precision and recall; useful when classes are uneven.",
    "roc_auc": "Measures how well the model ranks the right class (especially for yes/no problems).",
}


def main() -> None:
    st.set_page_config(
        page_title="Hyperparameter helper",
        page_icon="🎛️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🎛️ Find better settings for your prediction model")
    st.markdown(
        "**In plain terms:** your model has knobs (settings). This tool tries many combinations "
        "and shows which ones work best on your data — so you do not have to guess."
    )

    with st.expander("What is “hyperparameter tuning”?", expanded=False):
        st.markdown(
            """
- A **model** learns patterns from rows of data (features) to predict a **label** (e.g. sick vs healthy).
- **Hyperparameters** are choices you make *before* training (e.g. how many trees, how deep).
- This app **searches** those choices using cross-validation: it trains on part of the data,
  checks quality on held-out folds, and reports the best mix it found.

You can use a **demo dataset** or **upload your own CSV** (one column must be the answer to predict).
            """
        )

    with st.sidebar:
        st.header("1. Data")
        source = st.radio(
            "Where is your data?",
            ["Use a demo dataset", "Upload a CSV file"],
            horizontal=False,
        )

        dataset_key = "breast_cancer"
        upload = None
        target_col: str | None = None

        if source == "Use a demo dataset":
            dataset_key = st.selectbox(
                "Choose a demo",
                options=["breast_cancer", "iris", "wine"],
                format_func=lambda k: {
                    "breast_cancer": "Breast cancer (realistic)",
                    "iris": "Iris flowers (small / fast)",
                    "wine": "Wine types",
                }[k],
                help="Built-in examples so you can try the tool without preparing a file.",
            )
            st.caption(DATASET_HELP[dataset_key])
        else:
            upload = st.file_uploader(
                "CSV file",
                type=["csv"],
                help="Table with one column to predict (class labels). Other columns are inputs.",
            )
            target_col = st.text_input(
                'Name of the column to predict (optional)',
                placeholder="Leave empty to use the last column",
                help="If empty, the rightmost column is used as the label to predict.",
            )
            if target_col and not target_col.strip():
                target_col = None
            elif target_col:
                target_col = target_col.strip()

        st.header("2. Model & goal")
        model_key = st.selectbox(
            "Type of model",
            options=sorted(MODELS.keys()),
            format_func=lambda k: {
                "random_forest": "Random Forest",
                "gradient_boosting": "Gradient Boosting",
                "logistic_regression": "Logistic Regression",
                "svc_rbf": "Support Vector Machine (RBF)",
            }[k],
        )
        st.caption(MODEL_HELP[model_key])

        metric_key = st.selectbox(
            "How to measure “good”",
            options=["accuracy", "f1_macro", "roc_auc"],
            format_func=lambda k: {
                "accuracy": "Accuracy (simple)",
                "f1_macro": "F1 score (classes uneven)",
                "roc_auc": "ROC AUC (ranking quality)",
            }[k],
        )
        st.caption(METRIC_HELP[metric_key])

        method_key = st.selectbox(
            "Search style",
            options=["random", "grid", "optuna"],
            format_func=lambda k: {
                "random": "Random search (recommended)",
                "grid": "Grid search (exhaustive)",
                "optuna": "Optuna (smarter, slower)",
            }[k],
        )
        st.caption(METHOD_HELP[method_key])

        with st.expander("Advanced options"):
            cv = st.slider("Validation folds", 3, 10, 5, help="More folds = more reliable score, slower.")
            test_frac = st.slider(
                "Hold out test fraction",
                min_value=0.0,
                max_value=0.5,
                value=0.0,
                step=0.05,
                help="Optionally set aside some rows for a final score after tuning.",
            )
            if method_key == "random":
                n_iter = st.number_input("Random tries", 8, 200, 32, step=4)
            else:
                n_iter = 32
            if method_key == "optuna":
                n_trials = st.number_input("Optuna trials", 5, 200, 40, step=5)
            else:
                n_trials = 40
            seed = st.number_input("Random seed", 0, 2**31 - 1, 42)

        run_btn = st.button("Run search", type="primary", use_container_width=True)

    if not run_btn:
        st.info("Choose options in the sidebar and click **Run search**.")
        return

    if source == "Upload a CSV file" and upload is None:
        st.warning("Please upload a CSV file first.")
        return

    csv_bytes = upload.getvalue() if upload is not None else None

    try:
        with st.spinner("Searching settings… this may take a minute."):
            X_train, X_hold, y_train, y_hold = load_data(
                csv_path=None,
                csv_bytes=csv_bytes,
                dataset=dataset_key,
                target=target_col,
                test_size=float(test_frac),
                random_state=int(seed),
            )
            result = run(
                model_key,
                method_key,
                X_train,
                y_train,
                cv=int(cv),
                metric=metric_key,
                random_state=int(seed),
                n_iter=int(n_iter),
                n_trials=int(n_trials),
                n_jobs=1,
            )
            if X_hold is not None and y_hold is not None:
                est = result.get("_best_estimator")
                if est is not None:
                    result["holdout_score"] = holdout_score(est, X_hold, y_hold, metric_key)
    except Exception as e:
        st.error(f"Something went wrong: {e}")
        return

    out = public_result(result)

    st.subheader("Results")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            "Best score (cross-validation)",
            f"{out['best_cv_score']:.4f}",
            help="Average quality across held-out slices of the training data.",
        )
    with c2:
        st.metric("Model", out["model"].replace("_", " "))
    with c3:
        if "holdout_score" in out:
            st.metric("Score on held-out test slice", f"{out['holdout_score']:.4f}")
        else:
            st.metric("Held-out test", "—", help="Enable “Hold out test fraction” in Advanced to get this.")

    st.markdown("**Best settings found** (you can reuse these when training elsewhere):")
    st.json(out["best_params"])

    if "top_features" in out:
        st.markdown("**Which inputs mattered most** (for tree-based models):")
        imp_df = pd.DataFrame(out["top_features"])
        st.bar_chart(imp_df.set_index("feature")["importance"])

    with st.expander("Export JSON"):
        st.code(json.dumps(out, indent=2), language="json")


if __name__ == "__main__":
    main()

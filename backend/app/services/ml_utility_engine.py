"""
DataNova ML utility engine.

Runs the standard two-experiment TRTR/TSTR-style comparison:

  Experiment A (baseline): train model on REAL training rows -> test on REAL test rows
  Experiment B (utility):  train model on SYNTHETIC rows     -> test on the SAME REAL test rows

Comparing B against A answers: "if I train on synthetic data, how much ML
performance do I keep compared to training on real data?"

Important interpretation note surfaced to users: this experiment measures
downstream ML *usefulness*. It says nothing about privacy by itself.

Data handling:
- The real dataset is split once (stratified for classification) so both
  experiments are evaluated on exactly the same real test set (no leakage:
  the synthetic data is generated independently of the test rows).
- Categorical features are one-hot encoded with sklearn's ColumnTransformer;
  numerical features are passed through (trees) — no scaling needed for the
  baseline models we offer.
- Training rows are capped for speed on large datasets.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .. import config
from .dataset_analyzer import _infer_semantic_type
from .model_recommender import _model_defs


def _build_preprocessor(df: pd.DataFrame, feature_cols: List[str]) -> ColumnTransformer:
    """One-hot categoricals (handle unknowns), impute numerics with the median."""
    num_cols = [c for c in feature_cols if _infer_semantic_type(df[c]) == "numerical"]
    cat_cols = [c for c in feature_cols if _infer_semantic_type(df[c]) in ("categorical", "boolean")]

    num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01)),
    ])
    return ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols),
    ], remainder="drop")


def _classification_metrics(y_true, y_pred) -> Dict[str, Any]:
    """Accuracy, macro precision/recall/F1 + confusion matrix."""
    labels = sorted(set(y_true) | set(y_pred), key=str)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "confusion_matrix": {
            "labels": [str(l) for l in labels],
            "matrix": cm.astype(int).tolist(),
        },
    }


def _regression_metrics(y_true, y_pred) -> Dict[str, Any]:
    """MAE, MSE, RMSE, R2 — all computed from actual predictions."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": round(float(mae), 4),
        "mse": round(float(mse), 4),
        "rmse": round(float(np.sqrt(mse)), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def run_ml_utility(real: pd.DataFrame, synthetic: pd.DataFrame, target: str, task: str,
                   model_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Run Experiment A and Experiment B with the chosen (or recommended) model.

    Returns metrics for both experiments plus a per-metric retention ratio
    (B / A) where meaningful.
    """
    if target not in real.columns:
        raise ValueError(f"Target column '{target}' not found in the real dataset.")
    if task not in ("classification", "regression"):
        raise ValueError(f"Unsupported task '{task}' for the ML utility test.")

    real = real.dropna(subset=[target]).copy()
    if real[target].nunique() < 2:
        raise ValueError("Target column has fewer than 2 distinct values; ML test is not meaningful.")

    # Features must exist in BOTH frames; anything missing from the synthetic
    # data (e.g. an excluded ID column) is skipped and reported.
    feature_cols = [c for c in real.columns if c != target and c in synthetic.columns]
    skipped = [c for c in real.columns if c != target and c not in synthetic.columns]
    if not feature_cols:
        raise ValueError("No feature columns are shared between the real and synthetic datasets.")

    # ----- shared real hold-out (same test set for both experiments) -----
    stratify = real[target] if task == "classification" else None
    try:
        real_train, real_test = train_test_split(
            real, test_size=config.ML_TEST_SIZE, random_state=config.ML_RANDOM_STATE,
            stratify=stratify,
        )
    except ValueError:
        # Very small/imbalanced classes: fall back to an unstratified split
        real_train, real_test = train_test_split(
            real, test_size=config.ML_TEST_SIZE, random_state=config.ML_RANDOM_STATE,
        )

    # Cap training sizes so the demo stays responsive
    if len(real_train) > config.ML_MAX_TRAIN_ROWS:
        real_train = real_train.sample(config.ML_MAX_TRAIN_ROWS, random_state=config.ML_RANDOM_STATE)
    synth_train = synthetic
    if len(synth_train) > config.ML_MAX_TRAIN_ROWS:
        synth_train = synth_train.sample(config.ML_MAX_TRAIN_ROWS, random_state=config.ML_RANDOM_STATE)

    y_test = real_test[target]

    # ----- model selection -----
    models = _model_defs(task)
    if model_key is None or model_key not in models:
        # Fall back to the first (default) model for the task
        model_key = next(iter(models))
    model = models[model_key]["factory"]()

    preprocessor = _build_preprocessor(pd.concat([real_train, synth_train], axis=0), feature_cols)

    def _fit_predict(train_X: pd.DataFrame, train_y: pd.Series):
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(train_X[feature_cols], train_y)
        return pipe.predict(real_test[feature_cols])

    # ----- Experiment A: real -> real -----
    pred_a = _fit_predict(real_train, real_train[target])
    metrics_a = _classification_metrics(y_test, pred_a) if task == "classification" else _regression_metrics(y_test, pred_a)

    # ----- Experiment B: synthetic -> real -----
    pred_b = _fit_predict(synth_train, synth_train[target])
    metrics_b = _classification_metrics(y_test, pred_b) if task == "classification" else _regression_metrics(y_test, pred_b)

    # ----- retention ratios (B relative to A) -----
    retention: Dict[str, Any] = {}
    if task == "classification":
        for m in ("accuracy", "precision", "recall", "f1"):
            a, b = metrics_a[m], metrics_b[m]
            retention[m] = round(b / a, 4) if a and a != 0 else None
    else:
        # For error metrics lower is better, so retention is A / B (clipped)
        for m in ("mae", "mse", "rmse"):
            a, b = metrics_a[m], metrics_b[m]
            retention[m] = round(min(a / b, 1.0), 4) if b and b != 0 else None
        r2a, r2b = metrics_a["r2"], metrics_b["r2"]
        retention["r2"] = round(r2b / r2a, 4) if r2a and r2a != 0 else None

    return {
        "task": task,
        "target": target,
        "model": {"key": model_key, "label": models[model_key]["label"]},
        "skipped_columns": skipped,
        "split": {
            "real_train_rows": int(len(real_train)),
            "real_test_rows": int(len(real_test)),
            "synthetic_train_rows": int(len(synth_train)),
            "test_size": config.ML_TEST_SIZE,
            "random_state": config.ML_RANDOM_STATE,
        },
        "experiment_a": {"description": "Trained on real data, tested on real test data", "metrics": metrics_a},
        "experiment_b": {"description": "Trained on synthetic data, tested on the same real test data", "metrics": metrics_b},
        "retention": retention,
        "interpretation_note": (
            "These results measure downstream ML utility only — how well models trained on "
            "synthetic data perform on real data. They do NOT measure privacy, and synthetic "
            "data is not automatically private, anonymous or unbiased."
        ),
    }

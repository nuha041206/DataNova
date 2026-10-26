"""
DataNova ML utility engine.

Runs the standard two-experiment TRTR/TSTR-style comparison:

  Experiment A (baseline): train model on REAL training rows -> test on REAL test rows
  Experiment B (utility):  train model on SYNTHETIC rows     -> test on the SAME REAL test rows

Comparing B against A answers: "if I train on synthetic data, how much ML
performance do I keep compared to training on real data?"

Leakage rules enforced here:
- The real dataset is split ONCE (stratified for classification); both
  experiments are evaluated on exactly the same real test rows.
- Experiment A's preprocessing is fit on the real TRAINING rows only.
- Experiment B's preprocessing is fit on the SYNTHETIC training rows only.
  (A shared preprocessor would leak real-data statistics into experiment B.)
- The test rows are never used for fitting anything.

Feature handling:
- Numerical columns: median imputation, no scaling (tree/linear baselines
  with one-hot categoricals do not require it; scaling is monotonic and would
  not change tree results anyway).
- Categorical/boolean columns: most-frequent imputation + one-hot encoding
  (handle_unknown="ignore" so synthetic-only categories never crash scoring).
- Datetime columns: converted to year / month / day / weekday numbers
  (row-wise, so it cannot leak).
- Free-text / name / email / ID-like columns are EXCLUDED from features and
  reported — one-hot encoding them would be meaningless and distance-based
  misuse of them would be worse.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

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

# Semantic types that must never become ML features
_NON_FEATURE_SEMANTICS = {
    "person_name", "email", "indian_phone", "indian_pincode",
    "customer_id", "generic_id", "address", "date_of_birth",
}


def _classify_features(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Split feature columns into (numeric, categorical, datetime, dropped-with-reason)."""
    num: List[str] = []
    cat: List[str] = []
    dt: List[str] = []
    dropped: List[str] = []
    for c in feature_cols:
        s = df[c]
        sem = _infer_semantic_type(s)
        if pd.api.types.is_datetime64_any_dtype(s) or sem in ("datetime", "date"):
            dt.append(c)
        elif sem == "numerical":
            num.append(c)
        elif sem in ("categorical", "boolean"):
            cat.append(c)
        elif sem in _NON_FEATURE_SEMANTICS:
            dropped.append(c)
        elif sem == "generic_text":
            # Low-cardinality text is a usable category; high-cardinality text
            # (names, free text, IDs) is not an ML feature.
            if s.nunique(dropna=True) <= max(20, int(0.05 * len(s))):
                cat.append(c)
            else:
                dropped.append(c)
        else:
            num.append(c)
    return num, cat, dt, dropped


def _add_datetime_features(df: pd.DataFrame, dt_cols: List[str]) -> pd.DataFrame:
    """Row-wise datetime -> numeric parts (no fitting involved, no leakage)."""
    out = df.copy()
    for c in dt_cols:
        parsed = pd.to_datetime(out[c], errors="coerce")
        out[f"{c}__year"] = parsed.dt.year
        out[f"{c}__month"] = parsed.dt.month
        out[f"{c}__day"] = parsed.dt.day
        out[f"{c}__weekday"] = parsed.dt.weekday
        out = out.drop(columns=[c])
    return out


def _make_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """Per-experiment preprocessor; MUST be fit on that experiment's training rows only."""
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
    labels = sorted(set(y_true) | set(y_pred), key=str)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "confusion_matrix": {"labels": [str(l) for l in labels], "matrix": cm.astype(int).tolist()},
    }


def _regression_metrics(y_true, y_pred) -> Dict[str, Any]:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": round(float(mae), 4),
        "mse": round(float(mse), 4),
        "rmse": round(float(np.sqrt(mse)), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def _imbalance_report(target_series: pd.Series, task: str) -> Optional[Dict[str, Any]]:
    """Honest class-balance information; never modifies the data."""
    if task != "classification":
        return None
    shares = target_series.value_counts(normalize=True, dropna=True)
    if len(shares) < 2:
        return None
    minority = float(shares.min())
    is_imbalanced = minority < 0.2 if len(shares) == 2 else minority < 0.1
    return {
        "class_distribution": {str(k): round(float(v), 4) for k, v in shares.items()},
        "minority_share": round(minority, 4),
        "is_imbalanced": is_imbalanced,
        "note": (
            f"Target is imbalanced: the smallest class holds {minority:.0%} of rows. "
            "Accuracy alone overstates performance — read precision/recall/F1 and the confusion matrix."
        ) if is_imbalanced else None,
    }


def run_ml_utility(real: pd.DataFrame, synthetic: pd.DataFrame, target: str, task: str,
                   model_key: Optional[str] = None) -> Dict[str, Any]:
    """Run Experiment A and B with the SAME model on the SAME real hold-out."""
    if target not in real.columns:
        raise ValueError(f"Target column '{target}' not found in the real dataset.")
    if task not in ("classification", "regression"):
        raise ValueError(f"Unsupported task '{task}' for the ML utility test.")

    real = real.dropna(subset=[target]).copy()
    if real[target].nunique() < 2:
        raise ValueError("Target column has fewer than 2 distinct values; the ML test is not meaningful.")
    if target not in synthetic.columns:
        raise ValueError(
            f"Target column '{target}' is missing from the synthetic dataset; "
            "regenerate the synthetic data before running the ML utility test."
        )
    if len(real) < 20:
        raise ValueError(
            f"The real dataset has only {len(real)} usable rows after removing missing targets; "
            "at least 20 are required for a train/test split."
        )

    # Features present in BOTH frames, target always excluded.
    feature_cols = [c for c in real.columns if c != target and c in synthetic.columns]
    skipped = [c for c in real.columns if c != target and c not in synthetic.columns]
    if not feature_cols:
        raise ValueError("No feature columns are shared between the real and synthetic datasets.")

    num_cols, cat_cols, dt_cols, dropped = _classify_features(pd.concat([real, synthetic], axis=0), feature_cols)
    usable = num_cols + cat_cols + dt_cols
    if not usable:
        raise ValueError(
            "None of the shared columns are usable ML features "
            "(only free-text/identifier columns were found)."
        )

    # ----- shared real hold-out (identical for both experiments) -----
    stratify = real[target] if task == "classification" else None
    try:
        real_train, real_test = train_test_split(
            real, test_size=config.ML_TEST_SIZE, random_state=config.ML_RANDOM_STATE, stratify=stratify)
    except ValueError:
        real_train, real_test = train_test_split(
            real, test_size=config.ML_TEST_SIZE, random_state=config.ML_RANDOM_STATE)

    if len(real_train) > config.ML_MAX_TRAIN_ROWS:
        real_train = real_train.sample(config.ML_MAX_TRAIN_ROWS, random_state=config.ML_RANDOM_STATE)
    synth_train = synthetic
    if len(synth_train) > config.ML_MAX_TRAIN_ROWS:
        synth_train = synth_train.sample(config.ML_MAX_TRAIN_ROWS, random_state=config.ML_RANDOM_STATE)

    # Synthetic rows without a target value cannot be trained on.
    synth_train = synth_train.dropna(subset=[target])
    if synth_train.empty:
        raise ValueError("Every synthetic row is missing the target value; ML utility cannot be evaluated.")

    # ----- datetime expansion + target typing (row-wise, leakage-free) -----
    real_train_x = _add_datetime_features(real_train, dt_cols)
    real_test_x = _add_datetime_features(real_test, dt_cols)
    synth_train_x = _add_datetime_features(synth_train, dt_cols)
    num_final = list(num_cols) + [f"{c}__{p}" for c in dt_cols for p in ("year", "month", "day", "weekday")]
    cat_final = list(cat_cols)

    if task == "classification":
        y_real_train = real_train[target].astype(str)
        y_synth_train = synth_train[target].astype(str)
        y_test = real_test[target].astype(str)
    else:
        y_real_train = pd.to_numeric(real_train[target], errors="coerce")
        keep = y_real_train.notna()
        real_train_x, y_real_train = real_train_x[keep], y_real_train[keep]
        y_synth_train = pd.to_numeric(synth_train[target], errors="coerce")
        keep_s = y_synth_train.notna()
        synth_train_x, y_synth_train = synth_train_x[keep_s], y_synth_train[keep_s]
        if synth_train_x.empty:
            raise ValueError("Synthetic target values are not numeric; cannot run the regression utility test.")
        y_test = pd.to_numeric(real_test[target], errors="coerce").fillna(0)

    models = _model_defs(task)
    if model_key is None or model_key not in models:
        model_key = next(iter(models))
    model = models[model_key]["factory"]()

    def _fit_predict(train_x: pd.DataFrame, train_y: pd.Series) -> np.ndarray:
        """Fresh preprocessor per experiment, fit on that experiment's training rows ONLY."""
        prep = _make_preprocessor(num_final, cat_final)
        pipe = Pipeline([("prep", prep), ("model", model)])
        pipe.fit(train_x[num_final + cat_final], train_y)
        return pipe.predict(real_test_x[num_final + cat_final])

    pred_a = _fit_predict(real_train_x, y_real_train)
    metrics_a = _classification_metrics(y_test, pred_a) if task == "classification" else _regression_metrics(y_test, pred_a)

    pred_b = _fit_predict(synth_train_x, y_synth_train)
    metrics_b = _classification_metrics(y_test, pred_b) if task == "classification" else _regression_metrics(y_test, pred_b)

    retention: Dict[str, Any] = {}
    if task == "classification":
        for m in ("accuracy", "precision", "recall", "f1"):
            a, b = metrics_a[m], metrics_b[m]
            retention[m] = round(b / a, 4) if a and a != 0 else None
    else:
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
        "dropped_columns": dropped,
        "features_used": usable,
        "imbalance": _imbalance_report(real[target], task),
        "split": {
            "real_train_rows": int(len(real_train)),
            "real_test_rows": int(len(real_test)),
            "synthetic_train_rows": int(len(synth_train)),
            "test_size": config.ML_TEST_SIZE,
            "random_state": config.ML_RANDOM_STATE,
        },
        "experiment_a": {"description": "Train on real → Test on real", "metrics": metrics_a},
        "experiment_b": {"description": "Train on synthetic → Test on the same real hold-out", "metrics": metrics_b},
        "retention": retention,
        "interpretation_note": (
            "Experiment A and B use the same model and the same real hold-out rows; only the "
            "training data differs (real vs synthetic). Preprocessing for each experiment is fit "
            "on its own training data only. These results measure downstream ML utility — not privacy."
        ),
    }

"""
DataNova model recommender service.

Suggests a baseline scikit-learn model from the detected task and simple data
characteristics. The recommendation is explainable and NOT presented as
universally best; the user can always override it in the UI.

Rules of thumb:
- Classification, mostly linear/low-interaction, small data -> LogisticRegression
- Classification, mixed types / suspected non-linear -> RandomForestClassifier
- Regression, mostly linear -> LinearRegression
- Regression, non-linear / mixed -> RandomForestRegressor

A DecisionTree option is offered as the most interpretable baseline.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from .dataset_analyzer import _infer_semantic_type


def _model_defs(task: str) -> Dict[str, Dict[str, Any]]:
    """Model catalogue with factory + human-readable strengths."""
    if task == "classification":
        return {
            "logistic_regression": {
                "label": "Logistic Regression",
                "factory": lambda: LogisticRegression(max_iter=1000),
                "strength": "Fast, highly interpretable linear baseline; works well when classes are roughly linearly separable.",
            },
            "decision_tree": {
                "label": "Decision Tree",
                "factory": lambda: DecisionTreeClassifier(max_depth=6, random_state=42),
                "strength": "Very interpretable rules; captures non-linear splits but can overfit.",
            },
            "random_forest": {
                "label": "Random Forest",
                # max_depth keeps small noisy datasets from overfitting
                "factory": lambda: RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1),
                "strength": "Robust non-linear model; handles mixed numerical/categorical (encoded) features with little tuning.",
            },
        }
    return {
        "linear_regression": {
            "label": "Linear Regression",
            "factory": lambda: LinearRegression(),
            "strength": "Interpretable linear baseline; strong when relationships are approximately linear.",
        },
        "decision_tree_regressor": {
            "label": "Decision Tree Regressor",
            "factory": lambda: DecisionTreeRegressor(max_depth=6, random_state=42),
            "strength": "Captures non-linear thresholds; easy to explain with tree plots.",
        },
        "random_forest_regressor": {
            "label": "Random Forest Regressor",
            "factory": lambda: RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1),
            "strength": "Robust non-linear regressor; handles mixed feature types with little tuning.",
        },
    }


def recommend_model(df: pd.DataFrame, task: str, target: str) -> Dict[str, Any]:
    """
    Choose a baseline model from data characteristics.

    Returns {"model_key", "label", "reason", "alternatives": [...]}.
    """
    if task not in ("classification", "regression") or target not in df.columns:
        return {
            "model_key": None,
            "label": None,
            "reason": "No usable target/task detected, so no baseline model can be recommended.",
            "alternatives": [],
        }

    feature_cols = [c for c in df.columns if c != target]
    n_num = sum(1 for c in feature_cols if _infer_semantic_type(df[c]) == "numerical")
    n_cat = sum(1 for c in feature_cols if _infer_semantic_type(df[c]) in ("categorical", "boolean"))
    n_rows = len(df)
    mixed = n_num > 0 and n_cat > 0
    mostly_categorical = n_cat > n_num

    models = _model_defs(task)

    if task == "classification":
        if n_rows < 300 or (n_num >= n_cat and not mostly_categorical and n_cat == 0):
            key = "logistic_regression"
            reason = (
                f"Logistic Regression was selected as the baseline because the dataset is small "
                f"({n_rows} rows) and predominantly numerical ({n_num} numerical features). "
                "It is a fast, interpretable linear baseline suited to this size of data."
            )
        elif mixed:
            key = "random_forest"
            reason = (
                f"Random Forest was selected as a baseline because the dataset contains mixed "
                f"feature relationships and both numerical ({n_num}) and encoded categorical ({n_cat}) variables. "
                "It captures non-linear interactions without heavy tuning."
            )
        else:
            key = "random_forest"
            reason = (
                f"Random Forest was selected as a robust default baseline for {n_num} numerical and "
                f"{n_cat} categorical features on {n_rows} rows; it typically performs well without tuning."
            )
    else:  # regression
        if mixed or n_cat > 0:
            key = "random_forest_regressor"
            reason = (
                f"Random Forest Regressor was selected because the data mixes {n_num} numerical and "
                f"{n_cat} categorical features, where non-linear effects are common."
            )
        else:
            key = "linear_regression"
            reason = (
                f"Linear Regression was selected because the features are purely numerical ({n_num} columns), "
                "making a linear baseline both fast and interpretable."
            )

    alts = [
        {"key": k, "label": v["label"], "strength": v["strength"]}
        for k, v in models.items() if k != key
    ]
    return {
        "model_key": key,
        "label": models[key]["label"],
        "reason": reason,
        "alternatives": alts,
        "available_models": {k: v["label"] for k, v in models.items()},
    }

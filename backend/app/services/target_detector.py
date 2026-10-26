"""
DataNova target & task detection service.

Given an analysed DataFrame, guess which column is most likely the target
(outcome) variable and whether the ML task is classification or regression.

The detection is heuristic and EXPLICITLY a suggestion: the API always returns
a confidence level, a score and a human-readable reason, and the UI lets the
user override both target and task.

Heuristics used (in rough priority order):
1. Column-name hints (target, label, class, churn, outcome, y, ...).
2. Binary / low-cardinality categorical columns (classic outcome shape).
3. Position: the last column is often the target in teaching datasets.
4. Columns are penalised if they look like IDs, are constant, or have very
   high cardinality.
5. Task type is derived from the chosen column's dtype and cardinality:
   numeric + many unique values -> regression, otherwise classification.
"""
from __future__ import annotations

import re

from typing import Any, Dict, List, Optional

import pandas as pd

from .dataset_analyzer import _infer_semantic_type, _looks_like_id

# Name hints with weights (strong names first)
_NAME_HINTS: Dict[str, float] = {
    "target": 0.5,
    "label": 0.45,
    "class": 0.4,
    "churn": 0.4,
    "outcome": 0.35,
    "approved": 0.3,
    "fraud": 0.3,
    "default": 0.3,
    "diagnosis": 0.3,
    "survived": 0.3,
    "passed": 0.3,
    "result": 0.3,
    "y": 0.25,
    "is_": 0.2,
    "has_": 0.2,
}

# Names that typically denote a continuous value being predicted (regression)
_REGRESSION_NAME_HINTS: Dict[str, float] = {
    "price": 0.3,
    "cost": 0.3,
    "revenue": 0.3,
    "salary": 0.3,
    "amount": 0.25,
    "sales": 0.25,
    "demand": 0.25,
    "temperature": 0.25,
}


def _confidence_from_score(score: float) -> str:
    if score >= 0.7:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


def detect_target_and_task(df: pd.DataFrame, override_target: Optional[str] = None) -> Dict[str, Any]:
    """
    Suggest a target column and task type.

    Returns dict with: suggested_target, task, confidence, score, reason,
    candidates (top alternatives with per-candidate reasons).
    """
    candidates: List[Dict[str, Any]] = []
    n_rows = len(df)

    for idx, col in enumerate(df.columns):
        series = df[col]
        reason_bits: List[str] = []
        score = 0.0

        inferred = _infer_semantic_type(series)
        unique = int(series.nunique(dropna=True))
        unique_ratio = unique / n_rows if n_rows else 0

        # Hard exclusions
        if _looks_like_id(str(col), series):
            continue
        if unique <= 1:
            continue
        if inferred == "datetime":
            # Datetime columns are almost never targets
            continue

        # 1) Name hints (classification-style first, then regression-style).
        # Short hints like 'y' must match a whole token ('y', 'y_val') and NOT
        # substrings ('city', 'years').
        lname = str(col).lower()
        tokens = re.split(r"[^a-z0-9]+", lname)
        matched = False
        for hint, weight in _NAME_HINTS.items():
            if len(hint) <= 2:
                hit = hint in tokens
            else:
                hit = hint in lname
            if hit:
                score += weight
                reason_bits.append(f"name '{col}' suggests an outcome variable (+{weight})")
                matched = True
                break
        if not matched:
            for hint, weight in _REGRESSION_NAME_HINTS.items():
                if hint in lname:
                    score += weight
                    reason_bits.append(f"name '{col}' suggests a value to predict (+{weight})")
                    break

        # 2) Low-cardinality categorical / boolean -> classic classification target
        if inferred in ("categorical", "boolean"):
            if unique == 2:
                score += 0.35
                reason_bits.append("binary categorical column, the typical shape of a classification target (+0.35)")
            elif unique <= 10:
                score += 0.25
                reason_bits.append(f"low-cardinality categorical column ({unique} categories) (+0.25)")
        elif inferred == "numerical":
            # Numeric columns: either a discrete class label (small cardinality)
            # or a continuous regression target.
            if unique == 2:
                score += 0.3
                reason_bits.append("binary numeric column, possible classification target (+0.3)")
            elif unique <= 10:
                score += 0.15
                reason_bits.append(f"numeric column with only {unique} distinct values, possibly a class label (+0.15)")
            elif unique > 10 and unique_ratio > 0.05:
                score += 0.2
                reason_bits.append(
                    f"continuous numeric column ({unique} distinct values), typical regression target (+0.2)"
                )

        # 3) Position bonus: last column slightly favoured (common convention)
        if idx == len(df.columns) - 1 and df.shape[1] > 1:
            score += 0.1
            reason_bits.append("is the last column, a common position for targets (+0.1)")

        # 4) Penalties: none for near-unique numeric columns any more —
        # continuous regression targets (price, income) are legitimately
        # almost-unique, and ID-like columns are already excluded above.

        score = max(0.0, score)
        if score > 0:
            task = _infer_task(series, inferred, unique)
            candidates.append({
                "column": str(col),
                "score": round(score, 3),
                "task": task,
                "reason": "; ".join(reason_bits) if reason_bits else "weak signal",
            })

    candidates.sort(key=lambda c: c["score"], reverse=True)

    if override_target and override_target in df.columns:
        # User override wins; describe it but mark confidence as user-set
        series = df[override_target]
        inferred = _infer_semantic_type(series)
        task = _infer_task(series, inferred, int(series.nunique(dropna=True)))
        return {
            "suggested_target": override_target,
            "task": task,
            "confidence": "user",
            "score": 1.0,
            "reason": f"Manually selected by the user. Detected as a {task} target.",
            "candidates": candidates[:5],
        }

    if not candidates:
        return {
            "suggested_target": None,
            "task": "unknown",
            "confidence": "low",
            "score": 0.0,
            "reason": "No column looked like a plausible target. You can select one manually.",
            "candidates": [],
        }

    best = candidates[0]
    return {
        "suggested_target": best["column"],
        "task": best["task"],
        "confidence": _confidence_from_score(best["score"]),
        "score": best["score"],
        "reason": best["reason"],
        "candidates": candidates[:5],
    }


def _infer_task(series: pd.Series, inferred: str, unique: int) -> str:
    """Decide classification vs regression for a chosen target column."""
    if inferred in ("categorical", "boolean"):
        return "classification"
    if inferred == "numerical":
        # Numeric with few distinct values -> treat as classification labels
        if unique <= 10:
            return "classification"
        return "regression"
    return "unknown"

"""
DataNova dataset analyzer service.

Reads an uploaded CSV, validates it and computes the *Dataset Health Check*:
row/column counts, missing values, duplicates, dtypes, cardinality, constant
columns, ID-like columns, class imbalance for a candidate target and a basic
statistical summary per column.

Design notes:
- Nothing is deleted or modified here. The health check is strictly read-only
  diagnostics; any cleaning decision is left to the user in the UI.
- All values returned are computed from the actual DataFrame.
"""
from __future__ import annotations

import io
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .. import config
from .types import ColumnProfile

# Columns that commonly hold row identifiers.
_ID_NAME_HINTS = ("id", "uuid", "guid", "index", "no", "number", "code", "key")

# Names that hint at an outcome/target column (used here only for a soft flag;
# full detection lives in the target_detector service).
_TARGET_NAME_HINTS = (
    "target", "label", "class", "churn", "outcome", "approved",
    "fraud", "default", "diagnosis", "survived", "passed", "result",
)


class DatasetError(ValueError):
    """Raised when an uploaded file cannot be used as a dataset."""


def load_csv_from_bytes(raw: bytes, filename: str) -> pd.DataFrame:
    """
    Parse uploaded bytes into a DataFrame with light validation.

    Raises DatasetError with a user-friendly message on failure.
    """
    if not filename.lower().endswith(".csv"):
        raise DatasetError("Only .csv files are supported.")

    if len(raw) == 0:
        raise DatasetError("The uploaded file is empty.")

    if len(raw) > config.MAX_UPLOAD_BYTES:
        raise DatasetError("File is larger than the 50 MB limit.")

    # utf-8 first, fall back to latin-1 (common with Excel-exported CSVs)
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = raw.decode("latin-1")
        except UnicodeDecodeError as exc:
            raise DatasetError("File must be UTF-8 or Latin-1 encoded text.") from exc

    try:
        df = pd.read_csv(io.StringIO(text))
    except pd.errors.EmptyDataError as exc:
        raise DatasetError("The CSV file contains no data rows.") from exc
    except pd.errors.ParserError as exc:
        raise DatasetError("Could not parse the CSV. Check the delimiter and header row.") from exc

    if df.empty:
        raise DatasetError("The CSV file contains no data rows.")
    if df.shape[1] < 2:
        raise DatasetError("Dataset needs at least 2 columns to be useful for synthesis.")

    # Normalise column names: strip whitespace, deduplicate
    df.columns = [str(c).strip() for c in df.columns]
    dupes = df.columns[df.columns.duplicated()].tolist()
    if dupes:
        raise DatasetError(f"Duplicate column names found: {dupes}")

    return df


def _infer_semantic_type(series: pd.Series) -> str:
    """Map a pandas Series to one of DataNova's semantic column types."""
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        # All integers with many repeats behave like categories; keep them numeric
        # unless cardinality is very low (<=2) where boolean-like handling is fine.
        return "numerical"
    # Try datetime parsing for object columns (sample to stay fast)
    if series.dropna().size and not pd.api.types.is_numeric_dtype(series):
        sample = series.dropna().astype(str).head(200)
        try:
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
            if parsed.notna().mean() >= 0.9:
                return "datetime"
        except Exception:
            pass
    return "categorical"


def _looks_like_id(name: str, series: pd.Series) -> bool:
    """
    Heuristic: does this column behave like a row identifier?

    Careful: continuous float columns (price, income, ...) are often unique but
    are real measures, NOT IDs. Only integer-like or name-hinted columns are
    treated as identifiers.
    """
    n = len(series)
    if n == 0:
        return False
    lname = name.lower().strip()
    name_hit = any(h in lname for h in ("id", "uuid", "guid", "key")) or lname in ("index", "no", "num")
    unique_ratio = series.nunique(dropna=True) / n
    if name_hit and unique_ratio > 0.95:
        return True
    # Sequential integer index columns (unique integer values). Floats are
    # excluded on purpose: a unique float column is usually a measure.
    if unique_ratio > 0.999 and pd.api.types.is_integer_dtype(series):
        return True
    return False


def profile_column(series: pd.Series) -> ColumnProfile:
    """Compute the health-check summary for a single column."""
    name = str(series.name)
    n = len(series)
    missing = int(series.isna().sum())
    unique = int(series.nunique(dropna=True))

    inferred = _infer_semantic_type(series)
    numeric_vals = pd.to_numeric(series, errors="coerce") if inferred == "numerical" else None

    prof = ColumnProfile(
        name=name,
        dtype=str(series.dtype),
        inferred_type=inferred,
        missing=missing,
        missing_pct=round(missing / n * 100, 2) if n else 0.0,
        unique=unique,
        unique_pct=round(unique / n * 100, 2) if n else 0.0,
    )

    if inferred == "numerical" and numeric_vals is not None:
        valid = numeric_vals.dropna()
        if len(valid):
            prof.mean = float(valid.mean())
            prof.std = float(valid.std()) if len(valid) > 1 else 0.0
            prof.min = float(valid.min())
            prof.max = float(valid.max())
            prof.q25 = float(valid.quantile(0.25))
            prof.median = float(valid.median())
            prof.q75 = float(valid.quantile(0.75))
    elif inferred in ("categorical", "boolean"):
        vc = series.value_counts(dropna=False).head(10)
        prof.top_values = [{"value": None if pd.isna(k) else _json_safe(k), "count": int(v)} for k, v in vc.items()]

    prof.is_constant = bool(unique <= 1)
    prof.is_high_cardinality = bool(unique > config.HIGH_CARDINALITY_THRESHOLD and inferred in ("categorical", "object"))
    prof.looks_like_id = _looks_like_id(name, series)
    return prof


def _json_safe(value: Any) -> Any:
    """Convert numpy/pandas scalars into plain JSON-safe Python values."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def analyze_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Full dataset health check.

    Returns a JSON-serialisable dict with dataset-level stats, per-column
    profiles and a list of human-readable health warnings.
    """
    n_rows, n_cols = df.shape

    # Whole-row duplicates (kept, only reported)
    duplicate_rows = int(df.duplicated().sum())

    # Suspicious values: numeric columns with negative values where the name
    # suggests a non-negative quantity (age, price, amount, ...).
    suspicious: List[Dict[str, Any]] = []
    non_negative_hints = ("age", "price", "amount", "income", "salary", "count", "quantity", "charge", "fee", "tenure")
    for col in df.columns:
        if _infer_semantic_type(df[col]) == "numerical":
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals) and vals.min() < 0 and any(h in col.lower() for h in non_negative_hints):
                suspicious.append({
                    "column": col,
                    "issue": "negative values",
                    "detail": f"Column '{col}' contains {int((vals < 0).sum())} negative values but its name suggests a non-negative quantity.",
                })

    # Constant / ID columns
    constant_cols = [str(c) for c in df.columns if df[c].nunique(dropna=True) <= 1]
    id_like_cols = [str(c) for c in df.columns if _looks_like_id(str(c), df[c])]

    # Class imbalance preview for every low-cardinality categorical column
    imbalance_info: Dict[str, Any] = {}
    for col in df.columns:
        inferred = _infer_semantic_type(df[col])
        if inferred in ("categorical", "boolean"):
            counts = df[col].value_counts(dropna=True)
            if 2 <= len(counts) <= 10:
                shares = (counts / counts.sum()).round(4)
                imbalance_info[str(col)] = {
                    "classes": len(counts),
                    "min_share": float(shares.min()),
                    "max_share": float(shares.max()),
                    "is_imbalanced": bool(shares.min() < 0.15),
                }

    columns: List[Dict[str, Any]] = []
    for col in df.columns:
        p = profile_column(df[col])
        d = p.__dict__.copy()
        # Soft target hint flag used by the UI
        d["target_name_hint"] = any(h in str(col).lower() for h in _TARGET_NAME_HINTS)
        columns.append(_json_safe(d))

    warnings: List[str] = []
    if n_rows < config.MIN_ROWS_RECOMMENDED:
        warnings.append(
            f"Small dataset ({n_rows} rows). SDV synthesizers learn distributions more reliably from at least ~{config.MIN_ROWS_RECOMMENDED} rows."
        )
    if duplicate_rows:
        warnings.append(f"{duplicate_rows} duplicate row(s) found. Consider removing them before generation (DataNova will not modify your data automatically).")
    if constant_cols:
        warnings.append(f"Constant column(s): {', '.join(constant_cols)}. They carry no information for ML models.")
    if id_like_cols:
        warnings.append(f"ID-like column(s) detected: {', '.join(id_like_cols)}. Synthetic IDs are meaningless and can be excluded from generation.")
    high_missing = [c["name"] for c in columns if c["missing_pct"] > 40]
    if high_missing:
        warnings.append(f"Column(s) with >40% missing values: {', '.join(high_missing)}.")
    for s in suspicious:
        warnings.append(s["detail"])
    for col, info in imbalance_info.items():
        if info["is_imbalanced"]:
            warnings.append(f"Column '{col}' is imbalanced (smallest class share {info['min_share']:.0%}).")

    return {
        "rows": int(n_rows),
        "columns": int(n_cols),
        "duplicate_rows": duplicate_rows,
        "total_missing_cells": int(df.isna().sum().sum()),
        "memory_mb": round(float(df.memory_usage(deep=True).sum() / 1024 / 1024), 2),
        "dtypes_summary": {str(k): int(v) for k, v in df.dtypes.astype(str).value_counts().items()},
        "column_profiles": columns,
        "constant_columns": constant_cols,
        "id_like_columns": id_like_cols,
        "imbalance": imbalance_info,
        "suspicious_values": suspicious,
        "warnings": warnings,
    }

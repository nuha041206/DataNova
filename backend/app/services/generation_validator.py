"""
DataNova generation validation ("Generation Quality Check").

Validates a generated DataFrame against the spec it was generated from:
row/column counts, technical dtypes, semantic validity (phone/email/pincode
formats), requested ranges, requested categories, missing values and
uniqueness. Produces a JSON report the UI shows before download; nothing is
silently accepted.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import pandas as pd

from .semantic_generators import generator_for
from .uniqueness import uniqueness_report

_PHONE_RE = re.compile(r"^[6-9]\d{9}$")
_PIN_RE = re.compile(r"^[1-9]\d{5}$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[a-zA-Z]{2,}$")

_SEMANTIC_FORMATS = {
    "indian_phone": _PHONE_RE,
    "indian_pincode": _PIN_RE,
    "email": _EMAIL_RE,
}


def validate_generation(
    df: pd.DataFrame,
    column_specs: List[Dict[str, Any]],
    n_requested: Optional[int] = None,
    unique_columns: Optional[List[str]] = None,
    original: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Build the Generation Quality Check report.

    column_specs: [{"name", "technical_type", "semantic_type", "min"?, "max"?,
                    "categories"?}, ...]
    Returns {passed, checks: [...], summary: {...}, uniqueness: {...}}
    """
    checks: List[Dict[str, Any]] = []
    ok = True

    def add(name: str, passed: bool, detail: str):
        nonlocal ok
        checks.append({"check": name, "passed": bool(passed), "detail": detail})
        ok = ok and passed

    # 1. row count
    if n_requested:
        add(
            "Row count",
            len(df) == n_requested,
            f"{len(df):,} rows generated" + (f" (requested {n_requested:,})" if len(df) != n_requested else ""),
        )

    # 2. column names
    expected_cols = [c["name"] for c in column_specs]
    if expected_cols:
        missing = [c for c in expected_cols if c not in df.columns]
        extra = [c for c in df.columns if c not in expected_cols]
        add("Column names", not missing and not extra,
            "All requested columns present" if not missing and not extra
            else f"missing: {missing}; unexpected: {extra}")

    # 3. per-column checks
    range_violations = 0
    category_violations = 0
    format_violations = 0
    dtype_violations = 0
    missing_cells = int(df.isna().sum().sum())

    for spec in column_specs:
        col = spec.get("name")
        if col not in df.columns:
            continue
        s = df[col]
        tech = spec.get("technical_type")
        sem = spec.get("semantic_type")

        # technical dtype
        actual = None
        if pd.api.types.is_integer_dtype(s):
            actual = "integer"
        elif pd.api.types.is_float_dtype(s):
            actual = "float"
        elif pd.api.types.is_bool_dtype(s):
            actual = "boolean"
        elif pd.api.types.is_datetime64_any_dtype(s):
            actual = "datetime"
        else:
            actual = "categorical" if s.nunique(dropna=True) <= max(20, len(s) * 0.05) else "string"
        if tech and actual != tech:
            compatible = {("integer", "float"), ("float", "integer"), ("string", "categorical"),
                          ("categorical", "string")}
            if (actual, tech) not in compatible:
                dtype_violations += 1
                add(f"Type of '{col}'", False, f"expected {tech}, got {actual}")

        # semantic format validity
        fmt = _SEMANTIC_FORMATS.get(sem)
        if fmt:
            vals = s.dropna().astype(str)
            bad = int((~vals.str.match(fmt)).sum())
            format_violations += bad
            add(f"Format of '{col}'", bad == 0,
                f"{bad} invalid {sem.replace('_', ' ')} value(s)" if bad
                else f"all {len(vals):,} values valid {sem.replace('_', ' ')} format")

        # requested range
        if spec.get("min") is not None or spec.get("max") is not None:
            nums = pd.to_numeric(s, errors="coerce").dropna()
            if len(nums):
                lo = spec.get("min")
                hi = spec.get("max")
                outside = int(((nums < lo) | (nums > hi)).sum()) if (lo is not None or hi is not None) else 0
                if outside:
                    range_violations += outside
                    add(f"Range of '{col}'", False, f"{outside} value(s) outside [{lo}, {hi}]")

        # requested categories
        if spec.get("categories"):
            allowed = set(spec["categories"])
            seen = set(s.dropna().astype(str))
            extra = seen - allowed
            if extra:
                category_violations += len(extra)
                add(f"Categories of '{col}'", False,
                    f"unexpected value(s): {sorted(extra)[:5]}")

    if range_violations == 0 and any(spec.get("min") is not None or spec.get("max") is not None for spec in column_specs):
        add("Requested ranges", True, "all values within requested ranges")
    if category_violations == 0 and any(spec.get("categories") for spec in column_specs):
        add("Requested categories", True, "all values come from the requested categories")
    # Missing-value rate: when a source dataset is provided, generation should
    # roughly PRESERVE its missingness (a source with gaps legitimately
    # produces synthetic gaps); only clearly excessive missingness fails.
    if original is not None and len(original):
        src_cells = max(original.shape[0] * original.shape[1], 1)
        syn_cells = max(len(df) * df.shape[1], 1)
        src_rate = float(original.isna().sum().sum()) / src_cells
        syn_rate = missing_cells / syn_cells
        if src_rate == 0:
            add("Missing values", missing_cells == 0,
                "no missing cells" if missing_cells == 0 else f"{missing_cells} missing cell(s)")
        else:
            add("Missing values", syn_rate <= src_rate + 0.10,
                f"{syn_rate:.1%} of cells missing (source dataset has {src_rate:.1%})")
    else:
        add("Missing values", missing_cells == 0,
            "no missing cells" if missing_cells == 0 else f"{missing_cells} missing cell(s)")

    # 4. duplicates + uniqueness
    uni = uniqueness_report(df, unique_columns)
    add("Duplicate rows", uni["duplicate_rows"] == 0,
        f"{uni['unique_rows']:,} unique of {uni['rows']:,} rows" if uni["duplicate_rows"] == 0
        else f"{uni['duplicate_rows']} duplicate row(s)")
    for col, info in uni["unique_columns"].items():
        add(f"Unique column '{col}'", info["is_unique"],
            f"{info['unique_values']:,} unique values" if info["is_unique"]
            else f"only {info['unique_values']:,} unique for {info['rows']:,} rows")

    # 5. overlap with original (Mode A)
    if original is not None and len(original):
        key_cols = [c for c in original.columns if c in df.columns]
        if key_cols:
            orig_keys = set(map(tuple, original[key_cols].astype(str).head(50000).itertuples(index=False)))
            overlap = sum(1 for r in df[key_cols].astype(str).itertuples(index=False) if tuple(r) in orig_keys)
            add("No copied source rows", overlap == 0,
                "no synthetic row equals a source row" if overlap == 0
                else f"{overlap} row(s) match source rows")

    return {
        "passed": ok,
        "checks": checks,
        "summary": {
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "unique_rows": uni["unique_rows"],
            "duplicate_rows": uni["duplicate_rows"],
            "missing_cells": missing_cells,
            "format_violations": format_violations,
            "range_violations": range_violations,
            "category_violations": category_violations,
            "dtype_violations": dtype_violations,
        },
        "uniqueness": uni,
    }

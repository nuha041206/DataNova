"""
DataNova messy-data generator.

Applies USER-SPECIFIED amounts of realistic messiness to a COPY of a dataset,
guided by each column's semantic type, so the result looks like real uncleaned
data instead of random noise:

    phone   -> format variants (+91 ..., 0..., grouped digits)
    date    -> format variants (12/01/2026, 12-Jan-2026, ...)
    name    -> whitespace + capitalization variants
    numeric -> occasional invalid entries ("twenty five", -2) and outliers
    any     -> missing values (NaN / empty string), duplicate rows

Hard rules:
- The input frame is NEVER modified; work happens on a copy.
- Every applied change is counted and reported as actual numbers.
- Percentages are targets: the summary reports what actually happened.
"""
from __future__ import annotations

import random
import re
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .dataset_analyzer import _infer_semantic_type

_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_NUM_WORDS = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
              6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
              11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
              20: "twenty", 25: "twenty five", 30: "thirty", 40: "forty", 50: "fifty",
              60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety", 100: "hundred"}


def _phone_variants(v: str, rng: random.Random) -> str:
    digits = re.sub(r"\D", "", v)
    if len(digits) == 10:
        return rng.choice([
            f"+91 {digits}", f"0{digits}", f"+91-{digits}",
            f"{digits[:5]} {digits[5:]}", f"({digits[:3]}) {digits[3:]}",
        ])
    return f" {v} " if rng.random() < 0.5 else v


def _date_variants(v: str, rng: random.Random) -> str:
    parsed = pd.to_datetime(v, errors="coerce")
    if pd.isna(parsed):
        return v
    d = parsed
    return rng.choice([
        d.strftime("%d/%m/%Y"), d.strftime("%d-%m-%Y"), d.strftime("%d-%b-%Y"),
        d.strftime("%d %B %Y"), d.strftime("%m/%d/%Y"), d.isoformat(),
    ])


def _email_variants(v: str, rng: random.Random) -> str:
    if "@" not in v:
        return v
    local, _, domain = v.partition("@")
    return rng.choice([
        f" {v}", f"{v} ", v.upper() if rng.random() < 0.5 else v,
        f"{local}..{domain}" if "." not in domain else v,
        f"{local}@ {domain}",
    ])


def _name_variants(v: str, rng: random.Random) -> str:
    return rng.choice([
        f" {v}", f"{v} ", f"  {v}  ",
        v.upper(), v.lower(),
        "".join(ch.upper() if rng.random() < 0.5 else ch.lower() for ch in v),
    ])


def _text_cap_variants(v: str, rng: random.Random) -> str:
    return rng.choice([v.upper(), v.lower(),
                       "".join(ch.upper() if rng.random() < 0.5 else ch.lower() for ch in v)])


def _whitespace_variants(v: str, rng: random.Random) -> str:
    return rng.choice([f" {v}", f"{v} ", f"  {v}  ", f"\t{v}"])


def _invalid_numeric(v: float, rng: random.Random) -> Any:
    """Occasional realistic invalid entries for numeric columns."""
    r = rng.random()
    if r < 0.4 and abs(v) <= 120 and float(v).is_integer():
        return _NUM_WORDS.get(int(v), str(int(v)))
    if r < 0.7:
        return -abs(v) if v >= 0 else v
    return v * 100 if abs(v) < 1000 else v / 100


def make_messy(df: pd.DataFrame, cfg: Dict[str, Any], seed: int = 42) -> Dict[str, Any]:
    """
    Apply the configured messiness to a copy of `df`.

    cfg keys (all optional floats 0..1 unless noted):
        missing_values, duplicate_rows, whitespace, capitalization,
        format_inconsistency, invalid_values, outliers
    Returns: {"dataframe", "summary", "config", "columns_affected"}
    """
    rng = random.Random(seed)
    out = df.copy()
    n_rows, n_cols = out.shape
    total_cells = n_rows * n_cols

    requested = {k: float(cfg.get(k) or 0) for k in
                 ("missing_values", "duplicate_rows", "whitespace", "capitalization",
                  "format_inconsistency", "invalid_values", "outliers")}
    summary: Dict[str, Any] = {
        "original_rows": int(n_rows),
        "original_columns": int(n_cols),
        "total_cells": int(total_cells),
        "requested": {k: round(v * 100, 2) for k, v in requested.items()},
    }
    columns_affected: Dict[str, List[str]] = {}

    sems = {c: _infer_semantic_type(out[c]) for c in out.columns}
    # Only exclude obvious unique identifiers from corruption. The name check
    # must NOT swallow columns like "Phone Number" (which legitimately receive
    # format variants) — only true ID-ish names (Customer ID, Employee Code).
    id_cols = {c for c in out.columns
               if out[c].is_unique and (sems[c] in ("customer_id", "generic_id")
                                        or bool(re.search(r"(?i)\b(id|uuid|guid|code)$", c)))}

    # ---------------- A. missing values ----------------
    target_cells = int(round(total_cells * requested["missing_values"]))
    applied_missing = 0
    if target_cells > 0:
        eligible = [c for c in out.columns if c not in id_cols]
        # Cast to object dtype first so NaN/"" can coexist with numbers
        # (matches how real uncleaned CSV data loads).
        for c in eligible:
            if not pd.api.types.is_object_dtype(out[c]):
                out[c] = out[c].astype(object)
        if eligible:
            per_col = max(1, target_cells // len(eligible))
            for c in eligible:
                k = min(per_col, max(1, int(len(out) * requested["missing_values"])))
                k = min(k, len(out))
                idx = rng.sample(range(len(out)), k)
                style = rng.random()
                for i in idx:
                    if pd.isna(out.at[out.index[i], c]):
                        continue
                    if style < 0.6:
                        out.iloc[out.index.get_loc(out.index[i]), out.columns.get_loc(c)] = np.nan
                    else:
                        out.iloc[out.index.get_loc(out.index[i]), out.columns.get_loc(c)] = ""
                    applied_missing += 1
                columns_affected.setdefault("missing", []).append(c)

    # ---------------- D/C. whitespace + capitalization (text-like columns) ----------------
    def _textish(c: str) -> bool:
        return sems[c] in ("generic_text", "person_name", "city", "state", "country",
                           "address", "email", "indian_phone", "category", "categorical",
                           "product_name", "company_name") or out[c].dtype == object

    def _apply_pct(c: str, fn) -> int:
        """Apply fn to ~pct of non-null cells of column c; return ACTUAL changed cells."""
        vals = out[c].dropna()
        if vals.empty:
            return 0
        k = int(round(len(vals) * pct))
        if k <= 0:
            return 0
        changed = 0
        for i in rng.sample(list(vals.index), min(k, len(vals))):
            cur = out.at[i, c]
            if isinstance(cur, str) and cur.strip():
                new = fn(cur, rng)
                if new != cur:
                    out.at[i, c] = new
                    changed += 1
        return changed

    pct = requested["whitespace"]
    ws_count = 0
    ws_cols: List[str] = []
    if pct > 0:
        for c in out.columns:
            if c not in id_cols and _textish(c) and out[c].notna().any():
                changed = _apply_pct(c, _whitespace_variants)
                if changed:
                    ws_cols.append(c)
                    ws_count += changed

    pct = requested["capitalization"]
    cap_count = 0
    cap_cols: List[str] = []
    if pct > 0:
        for c in out.columns:
            if c in id_cols or not _textish(c) or not out[c].notna().any():
                continue
            fn = _name_variants if sems[c] == "person_name" else _text_cap_variants
            cap_count += _apply_pct(c, fn)
            if c not in cap_cols and cap_count:
                cap_cols.append(c)
    summary["capitalization_issues"] = cap_cols

    # ---------------- E. format inconsistencies (semantic formats) ----------------
    def _format_kind(c: str) -> str | None:
        """Which format family a column belongs to, from semantic type OR name.

        Value-based detection alone returns 'categorical' for phone/email
        columns on small or constant datasets, so the column name is used as
        a second, honest signal.
        """
        sem = sems[c]
        if sem == "indian_phone":
            return "phone"
        if sem == "email":
            return "email"
        if sem in ("date", "datetime", "date_of_birth"):
            return "date"
        name_l = c.lower()
        if re.search(r"phone|mobile|contact|tel", name_l):
            return "phone"
        if re.search(r"e-?mail", name_l):
            return "email"
        if re.search(r"date|dob|birth", name_l):
            return "date"
        return None

    pct = requested["format_inconsistency"]
    fmt_count = 0
    fmt_cols: List[str] = []
    if pct > 0:
        for c in out.columns:
            if c in id_cols or not out[c].notna().any():
                continue
            kind = _format_kind(c)
            fn = {"phone": _phone_variants,
                  "date": _date_variants,
                  "email": _email_variants}.get(kind) if kind else None
            if fn is None:
                continue
            # Format variants are string forms ("+91 ...", "12-Jan-2026"), so
            # stringify numeric-stored values first — exactly like real
            # uncleaned CSV data where phone columns are mixed text.
            def _to_str(v: Any) -> Any:
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    return v
                if isinstance(v, float) and v.is_integer():
                    return str(int(v))
                if isinstance(v, (int, float)):
                    return str(v)
                return v
            out[c] = out[c].map(_to_str)
            changed = _apply_pct(c, fn)
            if changed:
                fmt_cols.append(c)
                fmt_count += changed
    summary["format_inconsistencies"] = fmt_cols

    # ---------------- F. invalid / noisy values (numeric) ----------------
    pct = requested["invalid_values"]
    inv_cols: List[str] = []
    if pct > 0:
        for c in out.columns:
            if c in id_cols:
                continue
            if sems[c] == "age":
                if not pd.api.types.is_object_dtype(out[c]):
                    out[c] = out[c].astype(object)  # allow "twenty five" / -2 next to numbers
                vals = out[c].dropna()
                if vals.empty:
                    continue
                k = max(1, int(round(len(vals) * pct)))
                for i in rng.sample(list(vals.index), min(k, len(vals))):
                    cur = out.at[i, c]
                    try:
                        cur_num = float(cur)
                    except (TypeError, ValueError):
                        continue
                    r = rng.random()
                    if r < 0.35:
                        out.at[i, c] = _NUM_WORDS.get(int(cur_num), str(int(cur_num)))
                    elif r < 0.6:
                        out.at[i, c] = -int(abs(cur_num)) if cur_num > 0 else cur_num
                    elif r < 0.8:
                        out.at[i, c] = 150
                    else:
                        out.at[i, c] = 0
                inv_cols.append(c)
            elif sems[c] == "numerical" and c not in ("salary",):
                if not pd.api.types.is_object_dtype(out[c]):
                    out[c] = out[c].astype(object)
                vals = out[c].dropna()
                if vals.empty:
                    continue
                k = max(1, int(round(len(vals) * pct)))
                for i in rng.sample(list(vals.index), min(k, len(vals))):
                    cur = out.at[i, c]
                    try:
                        out.at[i, c] = _invalid_numeric(float(cur), rng)
                    except (TypeError, ValueError):
                        continue
                inv_cols.append(c)
    summary["invalid_values"] = inv_cols

    # ---------------- G. outliers (numeric, controlled) ----------------
    pct = requested["outliers"]
    out_cols: List[str] = []
    if pct > 0:
        for c in out.columns:
            if c in id_cols or sems[c] not in ("numerical", "salary", "age"):
                continue
            vals = pd.to_numeric(out[c], errors="coerce").dropna()
            if len(vals) < 10:
                continue
            k = max(1, int(round(len(vals) * pct)))
            q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
            iqr = max(q3 - q1, 1e-9)
            for i in rng.sample(list(pd.to_numeric(out[c], errors="coerce").dropna().index), min(k, len(vals))):
                cur = pd.to_numeric(out.at[i, c], errors="coerce")
                if pd.isna(cur):
                    continue
                factor = rng.choice([3.5, -3.5])
                out.at[i, c] = cur + factor * iqr * rng.uniform(1.5, 3.0)
            out_cols.append(c)
    summary["outliers"] = out_cols

    # ---------------- B. duplicate rows (append, never overwrite) ----------------
    dup_count = 0
    if requested["duplicate_rows"] > 0 and len(out) > 0:
        dup_count = int(round(len(out) * requested["duplicate_rows"]))
        if dup_count > 0:
            src = rng.sample(list(out.index), min(dup_count, len(out)))
            dups = out.loc[src].copy()
            out = pd.concat([out, dups], ignore_index=True)

    messy = out
    dup_actual = int(messy.duplicated().sum())
    missing_actual = int(messy.isna().sum().sum())

    summary.update({
        "rows_after_duplicates": int(len(messy)),
        "duplicate_rows_added": dup_count,
        "duplicate_rows_total_present": dup_actual,
        "missing_cells_actual": missing_actual,
        "missing_cells_target": target_cells,
        "counts": {
            "missing_values": applied_missing,
            "duplicate_rows": dup_count,
            "whitespace": ws_count,
            "capitalization": cap_count,
            "format_inconsistencies": fmt_count,
            "invalid_values": sum(1 for _ in inv_cols) and int(round(len(out) * requested["invalid_values"])) or 0,
            "outliers": int(round(len(out) * requested["outliers"])) if out_cols else 0,
        },
    })
    return {"dataframe": messy, "summary": summary, "config": requested, "columns_affected": columns_affected}

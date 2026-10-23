"""
DataNova requirements-based synthesizer (Mode B engine).

Turns a confirmed RequirementSpec into a synthetic dataset using the SEMANTIC
generator system:

1. Relationship-aware pass: state -> city -> address -> pincode, name -> email,
   gender -> name, age -> salary (when both present). Each coordinated column
   receives the series it depends on through `context`.
2. Each column is then filled by its dedicated SemanticGenerator (Indian names,
   10-digit mobiles, PIN codes aligned to the state, example.com emails, ...).
3. ML target (when requested): computed from the generated features through a
   logistic/linear model so the utility test stays meaningful.
4. Uniqueness: dedupe + bounded regeneration until the requested count is
   unique (or an honest explanation).
5. Validation: the generation quality report is computed from the final frame.

Transparent rules and templates — no trained model is involved.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .types import RequirementSpec
from .semantic_generators import generator_for
from .uniqueness import enforce_unique_rows, enforce_unique_column
from .generation_validator import validate_generation

_DEFAULT_SEED = 42


def _rng(seed: Optional[int]) -> random.Random:
    return random.Random(seed if seed is not None else _DEFAULT_SEED)


# Ordered so dependencies are generated first (age before date_of_birth so
# dates of birth can follow the generated ages; pincode before address so
# addresses can append the row's PIN-code-shaped tail).
_DEPENDENCY_ORDER = [
    "gender", "state", "country", "city", "indian_pincode", "address",
    "person_name", "email", "age", "date_of_birth", "salary", "currency",
]


def _context_for(col: Dict[str, Any], series: Dict[str, List[Any]],
                  columns: List[Dict[str, Any]], country: Optional[str] = None) -> Dict[str, Any]:
    """Build the generator context for a column, wiring dependencies.

    Dependencies are found by MEANING (semantic_type), not by column name, so
    a column called "sex" still coordinates names and "province" still
    coordinates cities.
    """
    ctx: Dict[str, Any] = {
        "min": col.get("min"),
        "max": col.get("max"),
        "categories": col.get("categories"),
        "distribution": col.get("distribution"),
    }
    sem = col["semantic_type"]

    def _dep(semantic: str) -> Optional[str]:
        """Name of an already-generated column with this meaning, if any."""
        return next((c["name"] for c in columns
                     if c["semantic_type"] == semantic and c["name"] in series), None)

    gender_col = _dep("gender")
    if sem == "person_name":
        if gender_col:
            ctx["gender_series"] = series[gender_col]
        if country:
            ctx["country"] = country       # pick a country-appropriate name pool
    if sem == "email":
        name_col = _dep("person_name")
        if name_col:
            ctx["name_series"] = series[name_col]
    state_col = _dep("state")
    if sem == "city" and state_col:
        ctx["state_series"] = series[state_col]
    if sem == "address":
        city_col, state_col2 = _dep("city"), _dep("state")
        pin_col = _dep("indian_pincode")
        if city_col:
            ctx["city_series"] = series[city_col]
        if state_col2:
            ctx["state_series"] = series[state_col2]
        if pin_col:
            ctx["pincode_series"] = series[pin_col]
        if country:
            ctx["country"] = country       # non-Indian context -> generic style
    if sem == "indian_pincode" and state_col:
        ctx["state_series"] = series[state_col]
    age_col = _dep("age")
    if sem == "salary" and age_col:
        ctx["age_series"] = series[age_col]
    if sem == "date_of_birth" and age_col:
        # DOB follows the age column so the two stay consistent
        ctx["age_series"] = series[age_col]
    if sem == "country" and not ctx.get("categories") and country:
        ctx["categories"] = [country]
    if sem in ("customer_id", "generic_id") and not ctx.get("prefix"):
        ctx["prefix"] = "".join(ch for ch in str(col["name"])[:4].upper() if ch.isalpha()) or "ID"
    if sem == "gender" and not ctx.get("categories"):
        ctx["categories"] = ["Male", "Female", "Other"]
        ctx["weights"] = [49, 49, 2]
    return ctx


def _generate_columns(spec: RequirementSpec, n: int, rng: random.Random) -> pd.DataFrame:
    """Fill every column with its semantic generator, dependencies first."""
    columns = list(spec.columns or [])
    order = sorted(columns, key=lambda c: _DEPENDENCY_ORDER.index(c["semantic_type"])
                   if c["semantic_type"] in _DEPENDENCY_ORDER else 99)
    series: Dict[str, List[Any]] = {}
    for col in order:
        gen = generator_for(col["semantic_type"])
        ctx = _context_for(col, series, columns, country=spec.country)
        series[col["name"]] = gen.generate(n, rng, ctx)
    return pd.DataFrame(series)


def _validate_spec(spec: RequirementSpec) -> None:
    """Reject a broken schema with a clear message instead of a pandas error."""
    columns = list(spec.columns or [])
    names = [str(c.get("name") or "").strip() for c in columns]
    if any(not n for n in names):
        raise ValueError("Every column needs a name. Name it in the schema editor (or add it to the prompt).")
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(
            f"Duplicate column name(s) in the schema: {', '.join(dupes)}. "
            "Rename or remove them so each column is generated separately."
        )
    for col in columns:
        lo, hi = col.get("min"), col.get("max")
        if lo is not None and hi is not None and float(lo) > float(hi):
            raise ValueError(
                f"Column '{col['name']}': minimum ({lo}) is greater than maximum ({hi}). "
                "Swap them in the schema editor."
            )


def _add_unique_ids(df: pd.DataFrame, spec: RequirementSpec, rng: random.Random) -> List[str]:
    """Give ID-semantic columns unique values; return notes."""
    notes: List[str] = []
    for col in spec.columns or []:
        if col["semantic_type"] in ("customer_id", "generic_id") and col["name"] in df.columns:
            prefix = "".join(ch for ch in col["name"][:4].upper() if ch.isalpha()) or "ID"
            df, note = enforce_unique_column(
                df, col["name"],
                lambda k, _p=prefix: generator_for("customer_id").generate(
                    k, rng, {"prefix": _p, "width": max(5, len(str(len(df))) + 1)}))
            if note:
                notes.append(note)
    return notes


def _logistic_target(features: pd.DataFrame, rng: random.Random,
                     positive_pct: Optional[float]) -> np.ndarray:
    n = len(features)
    logits = np.zeros(n)
    for col in features.columns:
        s = features[col]
        if pd.api.types.is_numeric_dtype(s):
            std = s.std() or 1.0
            logits += ((s - s.mean()) / std).to_numpy() * rng.uniform(0.5, 1.5) * rng.choice([-1, 1])
        else:
            codes = pd.Categorical(s).codes
            effects = rng.normal(0, 1.0, size=max(int(codes.max()) + 1, 1))
            logits += effects[np.clip(codes, 0, len(effects) - 1)] * rng.choice([-1, 1])
    logits += np.array([rng.gauss(0, 0.8) for _ in range(n)])
    if positive_pct is not None and 0 < positive_pct < 100:
        threshold = np.percentile(logits, 100 - positive_pct)
        return (logits >= threshold).astype(int)
    probs = 1 / (1 + np.exp(-logits))
    return (np.random.default_rng(7).random(n) < probs).astype(int)


def _regression_target(features: pd.DataFrame, rng: random.Random) -> np.ndarray:
    n = len(features)
    target = np.zeros(n)
    for col in features.columns:
        s = features[col]
        if pd.api.types.is_numeric_dtype(s):
            std = s.std() or 1.0
            target += ((s - s.mean()) / std).to_numpy() * rng.uniform(0.5, 2.0) * rng.choice([-1, 1])
        else:
            codes = pd.Categorical(s).codes
            effects = rng.normal(0, 1.0, size=max(int(codes.max()) + 1, 1))
            target += effects[np.clip(codes, 0, len(effects) - 1)]
    target += np.array([rng.gauss(0, 0.6) for _ in range(n)])
    target = target - target.min() + 1.0
    return np.round(target * 100.0, 2)


def _as_spec_dict(spec: RequirementSpec) -> List[Dict[str, Any]]:
    return list(spec.columns or [])


def generate_from_spec(spec: RequirementSpec, seed: Optional[int] = None) -> Dict[str, Any]:
    """Generate the dataset described by a confirmed spec (with validation)."""
    rng = _rng(seed)
    warnings: List[str] = list(spec.warnings or [])

    n_rows = int(spec.rows) if spec.rows else 2000
    if not spec.columns:
        raise ValueError("No columns in the spec. Add columns in the schema editor before generating.")
    _validate_spec(spec)

    # ---- 1. generate the feature columns --------------------------------
    df = _generate_columns(spec, n_rows, rng)

    # unique ID columns
    warnings.extend(_add_unique_ids(df, spec, rng))

    # ---- 2. ML target (optional) -----------------------------------------
    positive_pct = (spec.class_distribution or {}).get("positive_pct") if spec.class_distribution else None
    if spec.task in ("classification", "regression") and spec.target:
        # numeric encoding of text features for the target model
        enc = df.copy()
        for c in enc.columns:
            if not pd.api.types.is_numeric_dtype(enc[c]):
                enc[c] = pd.Categorical(enc[c].astype(str)).codes
        if spec.task == "regression":
            df[spec.target] = _regression_target(enc, rng)
        else:
            df[spec.target] = _logistic_target(enc, rng, positive_pct)
            if positive_pct is None:
                share = df[spec.target].mean()
                if share < 0.2 or share > 0.8:
                    warnings.append(
                        f"Positive-class share came out at {share:.0%}; no explicit distribution was requested."
                    )
        df = df[[c for c in df.columns if c != spec.target] + [spec.target]]

    # ---- 3. uniqueness ----------------------------------------------------
    unique_cols = [c["name"] for c in _as_spec_dict(spec) if c.get("unique")]
    def make_more(k: int) -> pd.DataFrame:
        more = _generate_columns(spec, k, rng)
        return more

    df, uni_report = enforce_unique_rows(df, original=None, n_requested=n_rows,
                                         make_more=make_more, unique_columns=unique_cols)
    if uni_report.get("notes"):
        warnings.extend(uni_report["notes"])

    # ---- 3b. restore the user's requested column order -------------------
    spec_order = [c["name"] for c in _as_spec_dict(spec) if c["name"] in df.columns]
    tail = [c for c in df.columns if c not in spec_order]          # e.g. target
    df = df[spec_order + tail]

    # ---- 4. validation ----------------------------------------------------
    validation = validate_generation(
        df,
        column_specs=_as_spec_dict(spec) + ([{"name": spec.target, "technical_type":
                                              "integer" if spec.task == "classification" else "float"}]
                                            if spec.target else []),
        n_requested=n_rows,
        unique_columns=unique_cols,
    )

    description = (
        f"Generated {len(df):,} rows from {len(spec.columns)} semantic column(s) "
        f"({'unique records' if spec.unique else 'duplicates removed'}), using dedicated semantic "
        "generators per column type. Rules-based synthesis, not a trained model."
    )
    return {
        "dataframe": df,
        "n_rows": int(len(df)),
        "warnings": warnings,
        "description": description,
        "uniqueness": uni_report,
        "validation": validation,
    }

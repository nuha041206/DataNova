"""
DataNova SDV generation engine.

Thin, modular wrapper around the SDV library so the rest of DataNova never
imports SDV directly. Responsibilities:

1. Build SDV single-table metadata from a DataFrame (+ optional ID exclusion).
2. Choose a sensible synthesizer based on dataset characteristics:
     - GaussianCopula (fast, statistically transparent, good for small data)
     - CTGAN (deep-learning model, better for complex non-linear distributions)
     - TVAE (autoencoder alternative to CTGAN)
3. Generate a small preview (before the user commits to full generation).
4. Generate the full synthetic dataset with error handling.

Everything is synchronous and sequential: this is a teaching project, not a
distributed system. Long operations run in FastAPI's threadpool via def endpoints.
"""
from __future__ import annotations

import time
import warnings
from typing import Any, Dict, List, Optional

import pandas as pd

# SDV imports are isolated to this module on purpose.
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer, TVAESynthesizer

from .. import config
from .dataset_analyzer import _infer_semantic_type

# Default epochs for neural synthesizers. Deliberately modest: demo
# datasets are small and we prefer "works on a laptop in minutes" over
# "trains all night".
_CTGAN_EPOCHS_DEFAULT = 120


def build_metadata(df: pd.DataFrame, exclude_columns: Optional[List[str]] = None) -> Metadata:
    """
    Build SDV metadata for a single table.

    SDV infers the sdtype (numerical / categorical / datetime) itself. If a
    column looks like a unique string identifier we mark it as an SDV "id"
    column so SDV synthesizes unique values instead of learning its pattern.
    """
    exclude = set(exclude_columns or [])
    work = df.drop(columns=[c for c in exclude if c in df.columns])
    metadata = Metadata.detect_from_dataframes({"table": work})

    # SDV auto-detects columns NAMED like known PII types (City, State, Name,
    # Email, Phone, ...) as its built-in Faker-backed sdtypes, which would
    # invent brand-new fake values ("Abigailhaven", "Alabama") instead of
    # reusing the categories that actually exist in the uploaded dataset.
    # Fix at the source: closed-set string columns become plain categoricals
    # (so SDV learns the REAL value pool), and numeric columns stay numerical
    # (so a numeric phone column keeps its dtype).
    _NAME_PII_SDTYPES = {"city", "state", "country", "name", "first_name", "last_name",
                         "email", "phone_number", "address", "street_address",
                         "administrative_unit", "zipcode", "date_of_birth"}
    try:
        table_cols = metadata.tables["table"].columns
        for col in list(table_cols):
            try:
                sdtype = table_cols[col].get("sdtype")
            except AttributeError:
                sdtype = table_cols[col]["sdtype"]
            if sdtype in _NAME_PII_SDTYPES:
                series = work[col]
                if pd.api.types.is_numeric_dtype(series):
                    metadata.update_column(table_name="table", column_name=col, sdtype="numerical")
                else:
                    nunique = int(series.nunique(dropna=True))
                    if nunique <= 200 and nunique <= 0.5 * max(len(series), 1):
                        metadata.update_column(table_name="table", column_name=col, sdtype="categorical")
    except Exception:
        # Metadata API differs across SDV versions; detection fix is best-effort.
        pass

    # Mark a unique non-numeric column as an id column (helps SDV produce
    # valid unique values instead of copying real ones).
    try:
        for col in work.columns:
            if work[col].is_unique and not pd.api.types.is_numeric_dtype(work[col]):
                metadata.update_column(table_name="table", column_name=col, sdtype="id")
                break
    except Exception:
        # Metadata API differs slightly across SDV versions; non-fatal.
        pass

    # Guard against ID over-reach. SDV (and the loop above) classifies any
    # unique column as an "id", which corrupts human-readable columns: a
    # unique integer Age becomes random 7-digit numbers and a unique Address
    # becomes "sdv-id-xxxxxx" tokens. Restore the proper sdtype for every
    # id-marked column that our semantic detector does NOT consider a real
    # identifier (customer_id / generic_id / indian_phone / indian_pincode).
    try:
        from .semantic_types import detect_semantic_type
        _ID_SEMANTICS = {"customer_id", "generic_id", "indian_phone", "indian_pincode"}
        table = metadata.tables["table"]
        table_cols = table.columns
        for col in list(table_cols):
            try:
                sdtype = table_cols[col].get("sdtype")
            except AttributeError:
                sdtype = table_cols[col]["sdtype"]
            if sdtype != "id":
                continue
            semantic = str((detect_semantic_type(work[col]) or {}).get("semantic_type") or "")
            if semantic in _ID_SEMANTICS:
                continue                      # genuine identifier: keep SDV id handling
            # This column is NOT an identifier (age, address, name, ...):
            # restore its real sdtype. SDV may also have made it the table's
            # primary key, which would now be invalid — clear that too.
            try:
                if getattr(table, "primary_key", None) == col:
                    metadata.remove_primary_key(table_name="table")
            except (AttributeError, ValueError, TypeError):
                pass
            series = work[col]
            if pd.api.types.is_numeric_dtype(series):
                metadata.update_column(table_name="table", column_name=col, sdtype="numerical")
            elif pd.api.types.is_datetime64_any_dtype(series):
                metadata.update_column(table_name="table", column_name=col, sdtype="datetime")
            else:
                nunique = int(series.nunique(dropna=True))
                if nunique <= 200 and nunique <= 0.5 * max(len(series), 1):
                    metadata.update_column(table_name="table", column_name=col, sdtype="categorical")
                else:
                    metadata.update_column(table_name="table", column_name=col, sdtype="text")
    except Exception:
        # Metadata API differs across SDV versions; detection fix is best-effort.
        pass

    return metadata


def choose_synthesizer(df: pd.DataFrame, override: Optional[str] = None) -> Dict[str, Any]:
    """
    Pick a synthesizer from dataset characteristics.

    Rules of thumb (explainable and deterministic):
    - Small datasets (< 2,000 rows), > 20% missing values, or datetime columns
      -> GaussianCopula: fast, robust and statistically transparent.
    - Larger mixed datasets -> CTGAN: models complex non-linear relationships.
    - The Advanced Mode panel can override this choice explicitly.
    """
    if override:
        return {
            "synthesizer": override,
            "reason": "Manually selected in Advanced Mode.",
        }

    n_rows = len(df)
    n_cols = max(df.shape[1], 1)
    missing_ratio = float(df.isna().sum().sum()) / (n_rows * n_cols)
    has_datetime = any(_infer_semantic_type(df[c]) == "datetime" for c in df.columns)
    n_numeric = sum(1 for c in df.columns if _infer_semantic_type(df[c]) == "numerical")
    n_cat = sum(1 for c in df.columns if _infer_semantic_type(df[c]) == "categorical")
    # High-cardinality categoricals make CTGAN's one-hot space explode while
    # the copula models them fine as marginal distributions.
    max_cat_card = max((int(df[c].nunique(dropna=True)) for c in df.columns
                        if _infer_semantic_type(df[c]) == "categorical"), default=0)

    reasons = []
    if n_rows < 2000:
        reasons.append(f"small dataset ({n_rows} rows)")
    if missing_ratio > 0.2:
        reasons.append(f"{missing_ratio:.0%} of cells are missing")
    if has_datetime:
        reasons.append("datetime column(s) are present")

    if reasons or max_cat_card > 50:
        if max_cat_card > 50 and not reasons:
            reasons.append(f"a categorical column has {max_cat_card} distinct values")
        return {
            "synthesizer": "GaussianCopula",
            "reason": (
                f"GaussianCopula was selected because {'; '.join(reasons)}. It models each column's "
                "distribution and the correlations between numeric columns, is fast, and is the "
                "most reliable choice for preserving the structure of the uploaded data."
            ),
            "dataset_profile": {
                "rows": n_rows, "columns": n_cols, "numeric": n_numeric,
                "categorical": n_cat, "max_categorical_cardinality": max_cat_card,
                "missing_ratio": round(missing_ratio, 3),
            },
        }

    if n_rows <= 20_000 and n_cat >= n_numeric:
        return {
            "synthesizer": "TVAE",
            "reason": (
                f"TVAE was selected because the dataset has {n_rows} rows with {n_cat} categorical vs "
                f"{n_numeric} numerical columns: autoencoder-based synthesis handles mixed categorical "
                "data well at this scale while training faster than CTGAN."
            ),
            "dataset_profile": {
                "rows": n_rows, "columns": n_cols, "numeric": n_numeric,
                "categorical": n_cat, "max_categorical_cardinality": max_cat_card,
                "missing_ratio": round(missing_ratio, 3),
            },
        }

    return {
        "synthesizer": "CTGAN",
        "reason": (
            f"CTGAN was selected because the dataset has {n_rows} rows with "
            f"{n_numeric} numerical and {n_cat} categorical columns of moderate cardinality. "
            "CTGAN models complex non-linear relationships well at this scale."
        ),
        "dataset_profile": {
            "rows": n_rows, "columns": n_cols, "numeric": n_numeric,
            "categorical": n_cat, "max_categorical_cardinality": max_cat_card,
            "missing_ratio": round(missing_ratio, 3),
        },
    }


def _build_synthesizer(name: str, metadata: Metadata, advanced: Optional[Dict[str, Any]] = None):
    """Instantiate the SDV synthesizer object from its short name."""
    if name == "CTGAN":
        epochs = int((advanced or {}).get("epochs", _CTGAN_EPOCHS_DEFAULT))
        return CTGANSynthesizer(metadata, epochs=epochs, verbose=False)
    if name == "TVAE":
        epochs = int((advanced or {}).get("epochs", _CTGAN_EPOCHS_DEFAULT))
        return TVAESynthesizer(metadata, epochs=epochs, verbose=False)
    return GaussianCopulaSynthesizer(metadata)


class FittedSynthesizer:
    """A synthesizer that has been fitted once and can be sampled repeatedly.

    Keeping the fitted model alive lets uniqueness top-up rounds sample more
    rows WITHOUT refitting — refitting CTGAN every round was the reason Mode A
    generation appeared stuck on the progress screen.
    """

    def __init__(self, df: pd.DataFrame, synthesizer_name: str, reason: str,
                 synthesizer: Any, exclude_columns: Optional[List[str]]):
        self.source = df
        self.approach = synthesizer_name
        self.reason = reason
        self._synth = synthesizer
        self.exclude_columns = exclude_columns or []
        self.fit_seconds: float = 0.0
        self.warnings: List[str] = []

    def sample(self, n_rows: int) -> pd.DataFrame:
        """Sample n rows and restore excluded ID columns to keep the schema."""
        synthetic = self._synth.sample(num_rows=int(max(n_rows, 1)))
        for col in self.exclude_columns:
            if col in self.source.columns and col not in synthetic.columns:
                synthetic[col] = [f"SYN{i + 1:06d}" for i in range(len(synthetic))]
        return synthetic


def structure_violations(synthetic: pd.DataFrame, source: pd.DataFrame,
                         skip_columns: Optional[List[str]] = None) -> Dict[str, Any]:
    """Compare a synthetic frame against the SOURCE dataset's own structure.

    Catches the failure mode where generation 'completed' but the data does
    not follow the upload: invented categories, out-of-range numerics, or a
    frame that is mostly duplicates. Returns counts the generate endpoint can
    use to decide on a fallback retry — and the UI can show honestly.

    `skip_columns`: columns deliberately REBUILT by the semantic generators
    (names, phones, pincodes, addresses, ...). Their new values are valid
    semantic values by construction, so flagging them as "invented categories"
    would be a false warning (and could trigger a pointless synthesizer retry).
    """
    violations: Dict[str, Any] = {
        "invented_category_values": 0,
        "out_of_range_numeric": 0,
        "duplicate_rate": 0.0,
        "details": [],
    }
    if synthetic is None or synthetic.empty:
        violations["details"].append("no data generated")
        return violations

    skip = set(skip_columns or [])
    for col in source.columns:
        if col not in synthetic.columns or col in skip:
            continue
        s_src, s_syn = source[col], synthetic[col]
        if pd.api.types.is_numeric_dtype(s_src):
            lo, hi = float(s_src.min()), float(s_src.max())
            span = (hi - lo) or 1.0
            # Allow 10% margin beyond the observed range (generation may
            # legitimately extrapolate slightly; wild magnitudes fail).
            tol = span * 0.10
            nums = pd.to_numeric(s_syn, errors="coerce").dropna()
            bad = int(((nums < lo - tol) | (nums > hi + tol)).sum())
            if bad:
                violations["out_of_range_numeric"] += bad
                violations["details"].append(
                    f"'{col}': {bad} value(s) outside the source range [{lo:.2f}, {hi:.2f}]")
        else:
            pool = set(s_src.dropna().astype(str).unique())
            if 1 < len(pool) <= 200:
                extra = set(s_syn.dropna().astype(str).unique()) - pool
                if extra:
                    violations["invented_category_values"] += len(extra)
                    violations["details"].append(
                        f"'{col}': {len(extra)} value(s) not present in the source "
                        f"(e.g. {sorted(extra)[:3]})")

    violations["duplicate_rate"] = round(float(synthetic.duplicated().mean()), 4)
    if violations["duplicate_rate"] > 0.5:
        violations["details"].append(
            f"{violations['duplicate_rate']:.0%} of rows are duplicates of each other")
    return violations


def fit_synthesizer(df: pd.DataFrame, exclude_columns: Optional[List[str]] = None,
                    synthesizer_override: Optional[str] = None,
                    advanced: Optional[Dict[str, Any]] = None) -> FittedSynthesizer:
    """Choose and fit a synthesizer ONCE; sample from it as many times as needed.

    Raises RuntimeError with a readable message when fitting fails.
    """
    warnings.simplefilter("ignore")

    work = df.drop(columns=[c for c in (exclude_columns or []) if c in df.columns])
    approach = choose_synthesizer(work, override=synthesizer_override)
    synth_name = approach["synthesizer"]

    try:
        metadata = build_metadata(work)
        synthesizer = _build_synthesizer(synth_name, metadata, advanced)
        t0 = time.time()
        synthesizer.fit(work)
        fit_seconds = round(time.time() - t0, 1)
    except Exception as exc:  # pragma: no cover - depends on SDV internals
        raise RuntimeError(f"SDV fitting failed with {synth_name}: {exc}") from exc

    fitted = FittedSynthesizer(df, synth_name, approach["reason"], synthesizer,
                               exclude_columns=exclude_columns)
    fitted.fit_seconds = fit_seconds
    return fitted


def validate_synthetics(synthetic: pd.DataFrame, original: pd.DataFrame,
                        skip_columns: Optional[List[str]] = None) -> List[str]:
    """
    Sanity checks on generated data. Never silently accepts broken output.

    `skip_columns`: columns deliberately REBUILT by the semantic generators —
    their fresh values are valid by construction, so "categories not present
    in the real data" would be a misleading warning for them.

    Returns a list of human-readable problems (empty list = all good).
    """
    problems: List[str] = []
    if synthetic is None or synthetic.empty:
        problems.append("Generation produced no rows.")
        return problems

    missing_cols = [c for c in original.columns if c not in synthetic.columns]
    if missing_cols:
        problems.append(f"Synthetic data is missing column(s): {', '.join(missing_cols)}.")

    skip = set(skip_columns or [])
    common = [c for c in original.columns if c in synthetic.columns and c not in skip]
    for col in common:
        real_unique = set(original[col].dropna().unique())
        synth_unique = set(synthetic[col].dropna().unique())
        # Categorical columns must not invent unseen categories
        inferred = _infer_semantic_type(original[col])
        if inferred in ("categorical", "boolean"):
            extra = synth_unique - real_unique
            if extra:
                shown = ", ".join(str(x) for x in list(extra)[:5])
                problems.append(f"Column '{col}' contains categories not present in the real data ({shown}).")

    dup_rate = synthetic.duplicated().mean()
    if dup_rate > 0.5:
        problems.append(f"{dup_rate:.0%} of synthetic rows are exact duplicates of each other.")

    return problems


def generate_full(df: pd.DataFrame, n_rows: int, exclude_columns: Optional[List[str]] = None,
                  synthesizer_override: Optional[str] = None,
                  advanced: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Fit the chosen synthesizer and sample n_rows synthetic records.

    exclude_columns lets the user drop ID-like columns before fitting.

    Returns dict: dataframe, approach, reason, fit_seconds, sample_seconds, warnings.
    Raises RuntimeError with a readable message when fitting fails.
    """
    warnings.simplefilter("ignore")

    work = df.drop(columns=[c for c in (exclude_columns or []) if c in df.columns])
    n_rows = int(min(max(n_rows, 1), config.MAX_ROWS_FULL_GENERATION))

    approach = choose_synthesizer(work, override=synthesizer_override)
    synth_name = approach["synthesizer"]

    try:
        metadata = build_metadata(work)
        synthesizer = _build_synthesizer(synth_name, metadata, advanced)

        t0 = time.time()
        synthesizer.fit(work)
        fit_seconds = round(time.time() - t0, 1)

        t0 = time.time()
        synthetic = synthesizer.sample(num_rows=n_rows)
        sample_seconds = round(time.time() - t0, 1)
    except Exception as exc:  # pragma: no cover - depends on SDV internals
        raise RuntimeError(
            f"SDV generation failed with {synth_name}: {exc}"
        ) from exc

    # Restore the excluded ID columns as simple synthetic sequences, so the
    # output schema matches the input schema.
    for col in (exclude_columns or []):
        if col in df.columns and col not in synthetic.columns:
            synthetic[col] = [f"SYN{i + 1:06d}" for i in range(len(synthetic))]

    warnings_out = validate_synthetics(synthetic, work)

    return {
        "dataframe": synthetic,
        "approach": synth_name,
        "reason": approach["reason"],
        "fit_seconds": fit_seconds,
        "sample_seconds": sample_seconds,
        "warnings": warnings_out,
        "n_rows_requested": n_rows,
        "n_rows_generated": int(len(synthetic)),
    }

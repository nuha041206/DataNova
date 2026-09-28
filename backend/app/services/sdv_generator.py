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

    if n_rows < 2000 or missing_ratio > 0.2 or has_datetime:
        reasons = [f"dataset has {n_rows} rows"]
        if missing_ratio > 0.2:
            reasons.append(f"{missing_ratio:.0%} of cells are missing")
        if has_datetime:
            reasons.append("datetime column(s) are present")
        return {
            "synthesizer": "GaussianCopula",
            "reason": (
                f"GaussianCopula was selected because {'; '.join(reasons)}. "
                "It is fast, statistically transparent and reliable on smaller datasets."
            ),
        }

    return {
        "synthesizer": "CTGAN",
        "reason": (
            f"CTGAN was selected because the dataset has {n_rows} rows with "
            f"{n_numeric} numerical and {n_cat} categorical columns. CTGAN models "
            "complex non-linear relationships well at this scale."
        ),
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


def validate_synthetics(synthetic: pd.DataFrame, original: pd.DataFrame) -> List[str]:
    """
    Sanity checks on generated data. Never silently accepts broken output.

    Returns a list of human-readable problems (empty list = all good).
    """
    problems: List[str] = []
    if synthetic is None or synthetic.empty:
        problems.append("Generation produced no rows.")
        return problems

    missing_cols = [c for c in original.columns if c not in synthetic.columns]
    if missing_cols:
        problems.append(f"Synthetic data is missing column(s): {', '.join(missing_cols)}.")

    common = [c for c in original.columns if c in synthetic.columns]
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

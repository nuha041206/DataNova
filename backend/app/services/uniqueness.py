"""
DataNova uniqueness engine.

Reusable helpers that guarantee (or honestly report) uniqueness of generated
records:

- `enforce_unique_rows`: deduplicates a generated frame and resamples/
  regenerates additional rows until the requested count of unique rows is
  reached (bounded retries), then reports what happened.
- `enforce_unique_column`: regenerates a column until its values are unique
  (bounded retries) — used for ID-like columns.
- `uniqueness_report`: full-row + per-column duplicate statistics for the UI.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

_MAX_RETRY_ROUNDS = 6        # bounded so generation always terminates
_MAX_ROWS_PER_ROUND = 50_000


def enforce_unique_rows(
    synthetic: pd.DataFrame,
    original: Optional[pd.DataFrame],
    n_requested: int,
    make_more: Callable[[int], pd.DataFrame],
    unique_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Ensure the synthetic frame has exactly `n_requested` unique rows.

    `make_more(k)` must produce k *fresh* rows (calls back into the generator).

    Returns (final_frame, report) where report contains counts for the UI:
        requested / generated / unique / duplicates_removed / collision_rounds /
        possible / notes
    """
    notes: List[str] = []
    rounds = 0

    # Drop duplicate rows within the synthetic data itself
    synthetic = synthetic.drop_duplicates(ignore_index=True)

    # Drop rows that also exist in the original dataset (Mode A overlap check)
    if original is not None and len(original):
        try:
            key_cols = [c for c in original.columns if c in synthetic.columns]
            if key_cols:
                orig_keys = set(map(tuple, original[key_cols].astype(str).itertuples(index=False)))
                mask = [tuple(r) for r in synthetic[key_cols].astype(str).itertuples(index=False)]
                keep = [i for i, key in enumerate(mask) if key not in orig_keys]
                removed = len(synthetic) - len(keep)
                if removed:
                    synthetic = synthetic.iloc[keep].reset_index(drop=True)
                    notes.append(f"{removed} row(s) identical to the source dataset were removed.")
        except Exception:
            notes.append("Could not compare against the source dataset for overlap.")

    # Resample until we have enough unique rows (bounded)
    while len(synthetic) < n_requested and rounds < _MAX_RETRY_ROUNDS:
        rounds += 1
        need = n_requested - len(synthetic)
        batch = make_more(max(int(need * 1.15), need + 5))
        batch = batch.drop_duplicates(ignore_index=True)
        if original is not None and len(original) and len(batch):
            key_cols = [c for c in original.columns if c in batch.columns]
            if key_cols:
                orig_keys = set(map(tuple, original[key_cols].astype(str).itertuples(index=False)))
                mask = [tuple(r) for a_key in [0] for r in batch[key_cols].astype(str).itertuples(index=False)]
                batch = batch.iloc[[i for i, key in enumerate(mask) if key not in orig_keys]]
        synthetic = pd.concat([synthetic, batch], ignore_index=True).drop_duplicates(ignore_index=True)

    possible = len(synthetic) >= n_requested
    if not possible:
        notes.append(
            f"Only {len(synthetic)} unique rows could be produced after {rounds} generation rounds "
            f"(requested {n_requested}). The column constraints are too restrictive for more unique "
            "combinations; relax a range or add a unique ID column."
        )

    synthetic = synthetic.head(n_requested).reset_index(drop=True)

    report: Dict[str, Any] = {
        "requested": int(n_requested),
        "generated": int(len(synthetic)),
        "unique_rows": int(len(synthetic.drop_duplicates())),
        "duplicate_rows": int(synthetic.duplicated().sum()),
        "collision_rounds": rounds,
        "uniqueness_fully_satisfied": bool(len(synthetic) == n_requested and synthetic.duplicated().sum() == 0),
        "notes": notes,
    }
    return synthetic, report


def enforce_unique_column(
    df: pd.DataFrame, column: str,
    regenerate: Callable[[int], List[Any]],
) -> Tuple[pd.DataFrame, str]:
    """
    Make `df[column]` unique by regenerating colliding values in place.

    Returns (frame, note). Note is "" when nothing was needed.
    """
    if column not in df.columns:
        return df, ""
    dup_mask = df[column].duplicated(keep=False)
    if not dup_mask.any():
        return df, ""
    n_dup = int(dup_mask.sum())
    fresh = regenerate(n_dup)
    df.loc[dup_mask, column] = fresh
    # one more pass for pathological cases
    if df[column].duplicated().any():
        fresh2 = regenerate(int(df[column].duplicated().sum()))
        df.loc[df[column].duplicated(), column] = fresh2
    return df, f"Column '{column}' had {n_dup} duplicate value(s); they were regenerated to be unique."


def uniqueness_report(df: pd.DataFrame, unique_columns: Optional[List[str]] = None) -> Dict[str, Any]:
    """Statistics for the Generation Quality Check panel."""
    unique_columns = unique_columns or []
    per_column = {}
    for col in unique_columns:
        if col in df.columns:
            per_column[col] = {
                "unique_values": int(df[col].nunique()),
                "rows": int(len(df)),
                "is_unique": bool(df[col].is_unique),
            }
    return {
        "rows": int(len(df)),
        "unique_rows": int(len(df.drop_duplicates())),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_pct": round(float(df.duplicated().mean()) * 100, 2),
        "unique_columns": per_column,
    }

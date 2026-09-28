"""
DataNova evaluation engine.

Compares a real DataFrame and a synthetic DataFrame and produces all numbers
shown in the UI / PDF report. EVERY value is computed from the two frames —
nothing is hardcoded or fabricated.

Comparisons produced:
- Column overlap + per-column statistical similarity (KS statistic for numeric,
  TVD for categorical) -> sdmetrics-style but implemented with scipy so students
  can read the code.
- Distribution comparison detail per column (mean/std/min/quantiles or
  category proportions).
- Correlation matrix comparison (Pearson for numeric, Cramer's V style for
  categorical pairs) summarised as mean absolute difference.
- Target distribution comparison (class proportions or target stats).
- A transparent "DataNova Similarity Score" = weighted mean of the component
  similarities. It is *not* an industry standard; it is defined here and its
  components are always shown alongside it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as sps

from .dataset_analyzer import _infer_semantic_type

# ---------------------------------------------------------------------------
# Component weights for the transparent DataNova Similarity Score.
# Deliberately simple and documented: users can see every component anyway.
# ---------------------------------------------------------------------------
SCORE_WEIGHTS = {
    "statistical": 0.30,   # per-column KS / TVD
    "distribution": 0.30,  # same per-column detail, emphasised separately
    "correlation": 0.25,   # structure between columns
    "target": 0.15,        # target column distribution
}


def _jsonify(value: Any) -> Any:
    """Convert numpy scalars to native Python for JSON serialisation."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _ks_similarity(real: pd.Series, synth: pd.Series) -> Dict[str, Any]:
    """
    Two-sample Kolmogorov-Smirnov test between real and synthetic values.

    Returns the KS statistic (0 = identical CDFs, 1 = disjoint) and a
    similarity = 1 - statistic, so 1.0 means identical distributions.
    """
    real_v = pd.to_numeric(real, errors="coerce").dropna()
    synth_v = pd.to_numeric(synth, errors="coerce").dropna()
    if len(real_v) < 5 or len(synth_v) < 5:
        return {"statistic": None, "p_value": None, "similarity": None}
    stat, pval = sps.ks_2samp(real_v, synth_v)
    return {"statistic": round(float(stat), 4), "p_value": _jsonify(pval), "similarity": round(1 - float(stat), 4)}


def _tvd_similarity(real: pd.Series, synth: pd.Series) -> Dict[str, Any]:
    """
    Total Variation Distance between categorical distributions.

    TVD = 0.5 * sum(|p_real - p_synth|) over all categories.
    similarity = 1 - TVD.
    """
    r_counts = real.value_counts(normalize=True, dropna=True)
    s_counts = synth.value_counts(normalize=True, dropna=True)
    all_cats = set(r_counts.index) | set(s_counts.index)
    if not all_cats:
        return {"tvd": None, "similarity": None}
    tvd = 0.5 * sum(abs(float(r_counts.get(c, 0.0)) - float(s_counts.get(c, 0.0))) for c in all_cats)
    missing = set(r_counts.index) - set(s_counts.index)
    return {
        "tvd": round(tvd, 4),
        "similarity": round(1 - tvd, 4),
        "missing_categories": [str(c) for c in missing][:10],
    }


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramer's V association for two categorical series (bias-corrected)."""
    table = pd.crosstab(x, y)
    if table.size == 0:
        return 0.0
    chi2 = sps.chi2_contingency(table)[0]
    n = table.to_numpy().sum()
    r, k = table.shape
    phi2 = chi2 / n
    phi2corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1)) if n > 1 else 0.0
    rcorr = r - (r - 1) ** 2 / (n - 1) if n > 1 else r
    kcorr = k - (k - 1) ** 2 / (n - 1) if n > 1 else k
    denom = min(rcorr, kcorr) - 1
    return float(np.sqrt(phi2corr / denom)) if denom > 0 else 0.0


def _association_matrix(df: pd.DataFrame, cols: List[str], types: Dict[str, str]) -> pd.DataFrame:
    """
    Mixed-type association matrix:
    - numeric vs numeric -> absolute Pearson correlation
    - anything else      -> Cramer's V
    """
    out = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols, dtype=float)
    for i, a in enumerate(cols):
        for j in range(i + 1, len(cols)):
            b = cols[j]
            try:
                if types.get(a) == "numerical" and types.get(b) == "numerical":
                    r = df[[a, b]].corr(method="pearson").iloc[0, 1]
                    val = 0.0 if pd.isna(r) else abs(float(r))
                else:
                    val = _cramers_v(df[a], df[b])
            except Exception:
                val = 0.0
            out.loc[a, b] = val
            out.loc[b, a] = val
    return out


def evaluate(real: pd.DataFrame, synthetic: pd.DataFrame,
             target: Optional[str] = None) -> Dict[str, Any]:
    """
    Main entry point: full real-vs-synthetic evaluation.

    Returns a JSON-serialisable dict with every component and the composite
    DataNova Similarity Score.
    """
    real_cols = [c for c in real.columns if c in synthetic.columns]
    only_real = [c for c in real.columns if c not in synthetic.columns]
    only_synth = [c for c in synthetic.columns if c not in real.columns]

    types: Dict[str, str] = {c: _infer_semantic_type(real[c]) for c in real_cols}

    # ------------------------------------------------------------------
    # 1) Per-column statistical + distribution comparison
    # ------------------------------------------------------------------
    column_results: List[Dict[str, Any]] = []
    for col in real_cols:
        t = types[col]
        entry: Dict[str, Any] = {"column": col, "type": t}

        if t == "numerical":
            entry["ks"] = _ks_similarity(real[col], synthetic[col])
            entry["similarity"] = entry["ks"]["similarity"]

            rv, sv = real[col].dropna(), synthetic[col].dropna()
            rv = pd.to_numeric(rv, errors="coerce")
            sv = pd.to_numeric(sv, errors="coerce")
            rv, sv = rv.dropna(), sv.dropna()

            def _q(a: pd.Series, qs) -> List[float]:
                return [round(float(a.quantile(q)), 4) if len(a) else None for q in qs]

            qs = [0.25, 0.5, 0.75]
            entry["stats"] = {
                "mean": {"real": _jsonify(rv.mean()), "synthetic": _jsonify(sv.mean())},
                "std": {"real": _jsonify(rv.std()), "synthetic": _jsonify(sv.std())},
                "min": {"real": _jsonify(rv.min()), "synthetic": _jsonify(sv.min())},
                "max": {"real": _jsonify(rv.max()), "synthetic": _jsonify(sv.max())},
                "quantiles": {"real": _q(rv, qs), "synthetic": _q(sv, qs)},
            }
        else:
            entry["tvd"] = _tvd_similarity(real[col], synthetic[col])
            entry["similarity"] = entry["tvd"]["similarity"]
            r_counts = real[col].value_counts(normalize=True, dropna=True).head(10)
            s_counts = synthetic[col].value_counts(normalize=True, dropna=True)
            entry["proportions"] = [
                {
                    "category": _jsonify(k),
                    "real_share": round(float(r_counts.get(k, 0.0)), 4),
                    "synthetic_share": round(float(s_counts.get(k, 0.0)), 4),
                }
                for k in r_counts.index
            ]

        column_results.append(entry)

    similarities = [c["similarity"] for c in column_results if c["similarity"] is not None]
    stat_similarity = float(np.mean(similarities)) if similarities else None

    # ------------------------------------------------------------------
    # 2) Correlation / association comparison (sampled for speed)
    # ------------------------------------------------------------------
    assoc_cols = [c for c in real_cols if types[c] in ("numerical", "categorical", "boolean")]
    # Cap to 12 columns to keep O(n^2) pairwise computations fast
    assoc_cols = assoc_cols[:12]
    corr_detail: Dict[str, Any] = {"compared_columns": assoc_cols}
    if len(assoc_cols) >= 2:
        n_sample = min(len(real), 1000)
        r_assoc = _association_matrix(real[assoc_cols].sample(n_sample, random_state=42), assoc_cols, types)
        s_assoc = _association_matrix(synthetic[assoc_cols].sample(min(len(synthetic), 1000), random_state=42), assoc_cols, types)
        diff = (r_assoc - s_assoc).abs()
        np.fill_diagonal(diff.values, 0.0)
        upper = diff.where(np.triu(np.ones(diff.shape), k=1).astype(bool))
        pair_diffs = upper.stack().dropna()
        corr_detail = {
            "compared_columns": assoc_cols,
            "real_matrix": [[round(float(v), 3) for v in row] for row in r_assoc.values],
            "synthetic_matrix": [[round(float(v), 3) for v in row] for row in s_assoc.values],
            "columns": assoc_cols,
            "mean_abs_difference": round(float(pair_diffs.mean()), 4) if len(pair_diffs) else 0.0,
            "max_abs_difference": round(float(pair_diffs.max()), 4) if len(pair_diffs) else 0.0,
            "largest_diffs": [
                {"pair": [str(a), str(b)], "real": round(float(r_assoc.loc[a, b]), 3),
                 "synthetic": round(float(s_assoc.loc[a, b]), 3),
                 "difference": round(float(d), 3)}
                for (a, b), d in pair_diffs.sort_values(ascending=False).head(5).items()
            ],
        }
    corr_similarity = max(0.0, 1 - corr_detail.get("mean_abs_difference", 0.0)) if len(assoc_cols) >= 2 else None

    # ------------------------------------------------------------------
    # 3) Target distribution comparison
    # ------------------------------------------------------------------
    def entry_proportions(r: pd.Series, s: pd.Series) -> List[Dict[str, Any]]:
        """Class shares for the same categories in both frames."""
        rc = r.value_counts(normalize=True, dropna=True)
        sc = s.value_counts(normalize=True, dropna=True)
        cats = list(rc.index) + [c for c in sc.index if c not in rc.index]
        return [
            {"category": _jsonify(k), "real_share": round(float(rc.get(k, 0.0)), 4),
             "synthetic_share": round(float(sc.get(k, 0.0)), 4)}
            for k in cats
        ]

    target_similarity: Optional[float] = None
    target_detail: Dict[str, Any] = {}
    if target and target in real.columns and target in synthetic.columns:
        rt = _infer_semantic_type(real[target])
        if rt == "numerical" and real[target].nunique(dropna=True) > 10:
            ks = _ks_similarity(real[target], synthetic[target])
            target_detail = {"kind": "regression", "ks": ks,
                             "stats": {
                                 "mean": {"real": _jsonify(pd.to_numeric(real[target]).mean()),
                                          "synthetic": _jsonify(pd.to_numeric(synthetic[target]).mean())},
                                 "std": {"real": _jsonify(pd.to_numeric(real[target]).std()),
                                         "synthetic": _jsonify(pd.to_numeric(synthetic[target]).std())},
                             }}
            target_similarity = ks["similarity"]
        else:
            tvd = _tvd_similarity(real[target], synthetic[target])
            target_detail = {"kind": "classification", "tvd": tvd,
                             "class_proportions": entry_proportions(real[target], synthetic[target])}
            target_similarity = tvd["similarity"]
    elif target:
        target_detail = {"kind": "unavailable", "reason": "target column missing from one of the datasets"}

    # ------------------------------------------------------------------
    # 4) Composite DataNova Similarity Score
    # ------------------------------------------------------------------
    components = {
        "statistical": stat_similarity,
        "distribution": stat_similarity,  # same per-column detail, shown separately in UI
        "correlation": corr_similarity,
        "target": target_similarity,
    }
    used = {k: v for k, v in components.items() if v is not None}
    score = None
    if used:
        total_w = sum(SCORE_WEIGHTS[k] for k in used)
        score = round(sum(SCORE_WEIGHTS[k] * v for k, v in used.items()) / total_w, 4)

    return {
        "data_nova_score": {
            "value": score,
            "definition": ("Weighted mean of the available component similarities "
                           "(statistical 30%, distribution 30%, correlation 25%, target 15%). "
                           "This is DataNova's own transparent composite metric, not an industry standard; "
                           "the individual components are always shown."),
            "components": components,
            "weights": SCORE_WEIGHTS,
        },
        "column_overlap": {"shared": real_cols, "only_in_real": only_real, "only_in_synthetic": only_synth},
        "column_results": column_results,
        "correlation_comparison": corr_detail,
        "target_comparison": {"target": target, "similarity": target_similarity, **target_detail},
        "row_counts": {"real": int(len(real)), "synthetic": int(len(synthetic))},
    }

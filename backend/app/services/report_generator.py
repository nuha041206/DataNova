"""
DataNova report generator service.

Renders the PDF evaluation report with ReportLab. The report is built ONLY from
the canonical project dict stored in SQLite — the same payloads the web UI
renders (analysis, detection, evaluation, ml_utility, explanation). Nothing is
recalculated and nothing is invented: charts are drawn from the stored summary
statistics (quantiles, category proportions, association matrices).

Pipeline:
    Project (SQLite) -> ReportData (this module) -> PDF

Sections:
    1  Overview                    6  ML utility test (when run)
    2  Original dataset            7  Methodology
    3  Generation requirements     8  Limitations
    4  Synthetic dataset           9  Files & exports
    5  Real vs synthetic (Mode A) 10  DataNova Score
"""
from __future__ import annotations

import base64
import io
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .. import config

APP_VERSION = "1.0.0"

# DataNova brand palette (darkest → cream)
_DARK = colors.HexColor("#400202")
_RED = colors.HexColor("#6D0808")
_MUTED = colors.HexColor("#757D6F")
_CREAM = colors.HexColor("#EEEAD7")
_CREAM_TINT = colors.HexColor("#F5F2E8")
_TEXT = colors.HexColor("#400202")
_GREY = colors.HexColor("#5c655c")

_ASSETS = Path(__file__).resolve().parent.parent / "assets"  # app/assets
LOGO_PATH = _ASSETS / "logo_trimmed.png"
LOGO_MARK_PATH = _ASSETS / "logo_mark.png"

PAGE_W, PAGE_H = A4
_MARGIN = 1.7 * cm
_CONTENT_W = PAGE_W - 2 * _MARGIN


def _styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="H1Brand", parent=styles["Title"], textColor=_DARK,
                              fontSize=24, leading=28, spaceAfter=2))
    styles.add(ParagraphStyle(name="H2Brand", parent=styles["Heading2"], textColor=_RED,
                              spaceBefore=16, spaceAfter=6))
    styles.add(ParagraphStyle(name="H3Brand", parent=styles["Heading3"], textColor=_DARK,
                              fontSize=10.5, spaceBefore=10, spaceAfter=4))
    styles.add(ParagraphStyle(name="Body", parent=styles["Normal"], fontSize=9.5, leading=13.5,
                              textColor=_TEXT))
    styles.add(ParagraphStyle(name="Small", parent=styles["Normal"], fontSize=8, leading=11,
                              textColor=_GREY))
    styles.add(ParagraphStyle(name="Caption", parent=styles["Normal"], fontSize=8, leading=11,
                              textColor=_GREY, spaceBefore=3))
    return styles


def _table(data: list, widths=None, header=True, font_size=8) -> Table:
    """Brand-styled table helper."""
    t = Table(data, colWidths=widths, hAlign="LEFT", repeatRows=1 if header else 0)
    style = [
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("TEXTCOLOR", (0, 0), (-1, -1), _TEXT),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#ddd8c8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _CREAM_TINT]),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), _RED),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
    t.setStyle(TableStyle(style))
    return t


def _info_table(rows: List[List[str]]) -> Table:
    """Two-column key/value table for project facts."""
    t = Table(rows, colWidths=[4.6 * cm, _CONTENT_W - 4.6 * cm], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TEXTCOLOR", (0, 0), (-1, -1), _TEXT),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#ddd8c8")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 0), (0, -1), _RED),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, _CREAM_TINT]),
    ]))
    return t


# ---------------------------------------------------------------------------
# Chart helpers — drawn ONLY from stored evaluation payloads
# ---------------------------------------------------------------------------

def _quantile_chart(col: Dict[str, Any]) -> Optional[Image]:
    """
    Grouped bars: real vs synthetic quantiles for a numeric column.

    Validates the stored quantiles first: they may contain None (a numeric
    column with no computable stats, e.g. all-missing values). Charting only
    when BOTH sides have at least 3 finite numbers at the same positions —
    never inventing values for missing entries.
    """
    stats = col.get("stats") or {}
    q = stats.get("quantiles") or {}
    real_all, synth_all = q.get("real") or [], q.get("synthetic") or []

    def _finite(v: Any) -> bool:
        try:
            return v is not None and float(v) == float(v)   # not None, not NaN
        except (TypeError, ValueError):
            return False

    pairs = [(float(r), float(s))
             for r, s in zip(real_all, synth_all)
             if _finite(r) and _finite(s)]
    if len(pairs) < 3:
        return None            # not enough VALID numbers to draw honestly
    real = [p[0] for p in pairs]
    synth = [p[1] for p in pairs]
    labels = ["min", "Q25", "median", "Q75", "max"][: len(pairs)]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    x = range(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(4.4, 2.2), dpi=200)
    ax.bar([i - w / 2 for i in x], real, w, label="Real", color="#400202")
    ax.bar([i + w / 2 for i in x], synth, w, label="Synthetic", color="#D9A441")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title(f"{col.get('column', '')} — quantiles", fontsize=8.5, color="#400202")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img = Image(buf, width=9.2 * cm, height=4.6 * cm)
    return img


def _category_chart(col: Dict[str, Any]) -> Optional[Image]:
    """Grouped bars: real vs synthetic category shares for a categorical column."""
    props = (col.get("proportions") or [])[:6]
    if len(props) < 2:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    labels = [str(p.get("category"))[:14] for p in props]
    real = [float(p.get("real_share") or 0) for p in props]
    synth = [float(p.get("synthetic_share") or 0) for p in props]

    x = range(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(4.4, 2.2), dpi=200)
    ax.bar([i - w / 2 for i in x], real, w, label="Real", color="#400202")
    ax.bar([i + w / 2 for i in x], synth, w, label="Synthetic", color="#D9A441")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=6.5, rotation=20, ha="right")
    ax.tick_params(axis="y", labelsize=7)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title(f"{col.get('column', '')} — category shares", fontsize=8.5, color="#400202")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img = Image(buf, width=9.2 * cm, height=4.6 * cm)
    return img


def _matrix_heatmap(matrix: List[List[float]], cols: List[str], title: str) -> Optional[Image]:
    """
    Association-matrix heatmap from the stored evaluation payload.

    Validates the matrix first: cells may be None/NaN (e.g. association with
    a constant column is undefined). Draws only a fully finite, square matrix.
    """
    if not matrix or not cols:
        return None
    n = len(cols)

    def _finite(v: Any) -> bool:
        try:
            return v is not None and float(v) == float(v)
        except (TypeError, ValueError):
            return False

    if len(matrix) != n or any(len(row) != n for row in matrix) \
            or not all(_finite(v) for row in matrix for v in row):
        return None            # incomplete or non-numeric: skip the chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    matrix = [[float(v) for v in row] for row in matrix]
    fig, ax = plt.subplots(figsize=(4.4, 3.9), dpi=200)
    im = ax.imshow(matrix, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(cols, fontsize=6, rotation=35, ha="right")
    ax.set_yticklabels(cols, fontsize=6)
    ax.set_title(title, fontsize=8.5, color="#400202")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img = Image(buf, width=9.6 * cm, height=8.5 * cm)
    return img


# ---------------------------------------------------------------------------
# Report data assembly — pure transformation of the stored project dict
# ---------------------------------------------------------------------------

def _fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "—"
    try:
        v = float(x)
        if v != v:  # NaN
            return "—"
        s = f"{v:.{nd}f}".rstrip("0").rstrip(".")
        return s if s not in ("", "-0") else "0"
    except (TypeError, ValueError):
        return str(x)


def _pct(x: Any) -> str:
    try:
        return f"{float(x) * 100:.1f}%"
    except (TypeError, ValueError):
        return "—"


def _target_interpretations(tgt: Dict[str, Any], evalr: Dict[str, Any]) -> List[str]:
    """Rule-based sentences computed from the stored evaluation numbers."""
    out: List[str] = []
    sim = tgt.get("similarity")
    kind = tgt.get("kind")
    if sim is not None and kind == "classification":
        out.append(
            f"The synthetic dataset preserved the target class proportions "
            f"{'closely' if sim >= 0.9 else 'reasonably' if sim >= 0.75 else 'only partially'} "
            f"(target similarity {_fmt(sim)})."
        )
    elif sim is not None:
        out.append(
            f"The synthetic target distribution tracks the real one with KS-based similarity {_fmt(sim)}."
        )
    return out


def _column_interpretations(evalr: Dict[str, Any]) -> List[str]:
    """Sentences about the weakest/strongest columns, from stored similarities."""
    out: List[str] = []
    cols = [c for c in (evalr.get("column_results") or []) if c.get("similarity") is not None]
    if not cols:
        return out
    ranked = sorted(cols, key=lambda c: c["similarity"])
    worst = ranked[0]
    best = ranked[-1]
    out.append(
        f"The synthetic distribution of “{worst['column']}” differs most from the original "
        f"(similarity {_fmt(worst['similarity'])})."
    )
    if best is not worst and best["similarity"] is not None:
        out.append(
            f"“{best['column']}” is reproduced almost exactly (similarity {_fmt(best['similarity'])})."
        )
    mean_sim = sum(c["similarity"] for c in cols) / len(cols)
    out.append(f"Average per-column similarity across {len(cols)} shared columns: {_fmt(mean_sim)}.")
    return out


def _ml_interpretations(mlr: Dict[str, Any]) -> List[str]:
    """Sentences comparing experiment A vs B, from stored metrics only."""
    out: List[str] = []
    a = (mlr.get("experiment_a") or {}).get("metrics") or {}
    b = (mlr.get("experiment_b") or {}).get("metrics") or {}
    task = mlr.get("task")
    if task == "classification":
        for metric, label in (("recall", "recall"), ("f1", "F1"), ("accuracy", "accuracy")):
            av, bv = a.get(metric), b.get(metric)
            if av is None or bv is None:
                continue
            diff = bv - av
            direction = "higher" if diff > 0 else ("similar" if abs(diff) < 0.02 else "lower")
            out.append(
                f"The synthetic-trained model achieved {direction} {label} than the real-trained baseline "
                f"({_fmt(bv)} vs {_fmt(av)})."
            )
            break
        ret = (mlr.get("retention") or {}).get("f1")
        if ret is not None:
            out.append(f"F1 retention (synthetic ÷ real trained): {_pct(ret)} of the real-data baseline.")
    elif task == "regression":
        ra, rb = a.get("r2"), b.get("r2")
        if ra is not None and rb is not None:
            diff = rb - ra
            direction = "higher" if diff > 0 else ("similar" if abs(diff) < 0.02 else "lower")
            out.append(
                f"The synthetic-trained model reached {direction} R² than the real-trained baseline "
                f"({_fmt(rb)} vs {_fmt(ra)})."
            )
        ret = (mlr.get("retention") or {}).get("r2")
        if ret is not None:
            out.append(f"R² retention (synthetic ÷ real trained): {_pct(ret)} of the real-data baseline.")
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_report(project: Dict[str, Any], out_path: Path) -> Path:
    """
    Render the PDF evaluation report for one stored project.

    `project` is the database dict with analysis / detection / evaluation /
    ml_utility / explanation payloads attached — the exact same objects the
    web UI renders. Sections degrade gracefully when data is absent.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()
    story: List[Any] = []

    analysis = project.get("analysis") or {}
    detection = project.get("detection") or {}
    evalr = project.get("evaluation") or {}
    mlr = project.get("ml_utility") or {}
    expl = project.get("explanation") or {}
    req_spec = project.get("requirement_spec") or {}
    synth_preview = project.get("synth_preview") or {}
    warnings = project.get("warnings") or []

    mode_label = {"upload": "Expanded Dataset (Mode A)",
                  "requirements": "From Requirements (Mode B)"}.get(
        project.get("mode"), str(project.get("mode", "—")))

    # ---------------------------------------------------------------- cover
    # [LOGO] DataNova / Synthetic Data Studio — locked row, logo never floats
    logo_cell: Any = ""
    if LOGO_PATH.exists():
        # logo_trimmed.png is 1115x890; keep the box on that aspect ratio so
        # the mark is never stretched (platypus Image has no preserveAspectRatio).
        logo_cell = Image(str(LOGO_PATH), width=2.4 * cm, height=2.4 * cm * 890 / 1115)
    brand_row = Table(
        [[logo_cell, [Paragraph("DataNova", styles["H1Brand"]),
                      Paragraph("Synthetic Data Studio", styles["H3Brand"])]]],
        colWidths=[3.0 * cm, _CONTENT_W - 3.0 * cm], hAlign="LEFT",
    )
    brand_row.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (0, 0), 0),
        ("RIGHTPADDING", (0, 0), (0, 0), 6),
    ]))
    story.append(brand_row)
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("PROJECT REPORT", styles["H1Brand"]))
    story.append(Spacer(1, 0.35 * cm))
    story.append(_info_table([
        ["Project name", str(project.get("name", "Unnamed project"))],
        ["Date", datetime.now().strftime("%Y-%m-%d %H:%M")],
        ["Generation mode", mode_label],
    ]))
    story.append(Spacer(1, 0.35 * cm))

    # 1. Overview
    story.append(Paragraph("1. Overview", styles["H2Brand"]))
    story.append(_info_table([
        ["Project name", str(project.get("name", "—"))],
        ["Created", str(project.get("created_at", "—"))],
        ["Generation mode", mode_label],
        ["Source dataset", str(project.get("dataset_filename") or "— (generated from requirements)")],
        ["Original rows", str(project.get("rows_real") or "—")],
        ["Original columns", str(project.get("columns") or "—")],
        ["Report generated", datetime.now().strftime("%Y-%m-%d %H:%M")],
        ["DataNova version", APP_VERSION],
    ]))

    # 2. Original dataset
    story.append(Paragraph("2. Original dataset", styles["H2Brand"]))
    if analysis:
        story.append(_info_table([
            ["Rows", str(analysis.get("rows", "—"))],
            ["Columns", str(analysis.get("columns", "—"))],
            ["Missing cells", str(analysis.get("total_missing_cells", 0))],
            ["Duplicate rows", str(analysis.get("duplicate_rows", 0))],
            ["Constant columns", ", ".join(analysis.get("constant_columns") or []) or "none"],
            ["ID-like columns", ", ".join(analysis.get("id_like_columns") or []) or "none"],
        ]))
        det_rows: List[List[str]] = [["Detected target", "—"], ["Detected task", "—"], ["Detection note", "—"]]
        if detection:
            det_rows = [
                ["Detected target", str(detection.get("suggested_target") or "—")],
                ["Detected task", str(detection.get("task") or "—")],
                ["Detection note", f"{detection.get('confidence', '—')} confidence — {detection.get('reason', '')}"],
            ]
        if project.get("target_overridden"):
            det_rows.append(["Override", "The user manually overrode the automatic suggestion."])
        story.append(Spacer(1, 0.25 * cm))
        story.append(_info_table(det_rows))
        warns = (analysis.get("warnings") or [])[:10]
        if warns:
            story.append(Paragraph("Health warnings", styles["H3Brand"]))
            for w in warns:
                story.append(Paragraph(f"• {w}", styles["Body"]))
    else:
        story.append(Paragraph(
            "No source dataset was uploaded — this project was generated from natural-language requirements, "
            "so there is no real-data health check.", styles["Body"]))
    if warnings:
        story.append(Paragraph("Generation warnings", styles["H3Brand"]))
        for w in warnings[:10]:
            story.append(Paragraph(f"• {w}", styles["Body"]))

    # 3. Generation requirements
    story.append(Paragraph("3. Generation requirements", styles["H2Brand"]))
    cfg_rows = [
        ["Synthesizer", str(project.get("synthesizer") or "—")],
        ["Why this method", str(project.get("generation_reason") or "—")],
        ["Rows requested", str(project.get("rows_requested") or "—")],
        ["Target", str(project.get("target") or "—")],
        ["Task", str(project.get("task") or "—")],
        ["Fit time (s)", str(_fmt(project.get("fit_seconds"), 2))],
        ["Sample time (s)", str(_fmt(project.get("sample_seconds"), 2))],
    ]
    if req_spec:
        feats = ", ".join(f.get("name", "?") for f in (req_spec.get("features") or []))
        cfg_rows += [
            ["User requirement", f"“{req_spec.get('raw_text', '')}”"],
            ["Requested features", feats or "—"],
            ["Constraints", "; ".join(req_spec.get("constraints") or []) or "none"],
        ]
    story.append(_info_table(cfg_rows))

    # 4. Synthetic dataset
    story.append(Paragraph("4. Synthetic dataset", styles["H2Brand"]))
    synth_rows_note = ""
    sp_rows = synth_preview.get("rows") if isinstance(synth_preview, dict) else None
    n_synth_cols = len(evalr.get("column_overlap", {}).get("shared", [])) if evalr else None
    if not n_synth_cols and analysis:
        n_synth_cols = analysis.get("columns")
    story.append(_info_table([
        ["Synthetic rows generated", str(project.get("rows_synthetic") or "—")],
        ["Synthetic columns", str(n_synth_cols or "—")],
        ["Dtypes (source dataset)", ", ".join(f"{k}: {v}" for k, v in (analysis.get("dtypes_summary") or {}).items()) or "—"],
    ]))
    if sp_rows:
        cols = list(sp_rows[0].keys())[:8]
        rows = [[str(c) for c in cols]] + [
            [str(r.get(c, ""))[:18] for c in cols] for r in sp_rows[:4]
        ]
        story.append(Spacer(1, 0.25 * cm))
        story.append(Paragraph("First rows of the generated dataset", styles["H3Brand"]))
        story.append(_table(rows, font_size=7.5))
        synth_rows_note = " (sample rows above are truncated for display)"
    if synth_rows_note:
        story.append(Paragraph(f"Sample values{synth_rows_note}.", styles["Caption"]))

    tgt = evalr.get("target_comparison") or {}

    # ---------------------------------------------------------------- page 2+
    story.append(PageBreak())

    # 5. Real vs synthetic comparison (Mode A only)
    story.append(Paragraph("5. Real vs synthetic comparison", styles["H2Brand"]))
    if evalr:
        score = (evalr.get("data_nova_score") or {}).get("value")
        if score is not None:
            story.append(Paragraph(
                f"DataNova Score: <b>{_fmt(score)}</b> — a project-defined weighted metric "
                "(see section 7). Not an industry standard.", styles["Body"]))
            story.append(Spacer(1, 0.15 * cm))

        overlap = evalr.get("column_overlap") or {}
        story.append(_info_table([
            ["Shared columns", str(len(overlap.get("shared", [])))],
            ["Only in real data", ", ".join(overlap.get("only_in_real", [])) or "none"],
            ["Only in synthetic data", ", ".join(overlap.get("only_in_synthetic", [])) or "none"],
            ["Rows compared", f"{(evalr.get('row_counts') or {}).get('real', '—')} real vs "
                              f"{(evalr.get('row_counts') or {}).get('synthetic', '—')} synthetic"],
        ]))
        story.append(Spacer(1, 0.25 * cm))

        # Detailed per-column tables: numeric (means/median/std/quantiles) and categorical (proportions)
        numeric_rows: List[List[str]] = [["Column", "Real mean", "Syn mean", "Real median", "Syn median",
                                          "Real std", "Syn std"]]
        categorical_rows: List[List[str]] = [["Column", "Category", "Real share", "Synthetic share"]]
        numeric_q_rows: List[List[str]] = [["Column", "Stat", "Real", "Synthetic"]]

        for c in evalr.get("column_results") or []:
            st = c.get("stats") or {}
            if st:
                def med(i: int) -> Any:
                    q = st.get("quantiles") or {}
                    r = (q.get("real") or [None] * 5)
                    s = (q.get("synthetic") or [None] * 5)
                    return (r[i] if len(r) > i else None, s[i] if len(s) > i else None)
                mean_r = (st.get("mean") or {}).get("real")
                mean_s = (st.get("mean") or {}).get("synthetic")
                std_r = (st.get("std") or {}).get("real")
                std_s = (st.get("std") or {}).get("synthetic")
                rm, sm = med(2)
                numeric_rows.append([
                    str(c.get("column")), _fmt(mean_r), _fmt(mean_s),
                    _fmt(rm), _fmt(sm), _fmt(std_r), _fmt(std_s),
                ])
                for label, idx in (("min", 0), ("Q25", 1), ("Q75", 3), ("max", 4)):
                    rv, sv = med(idx)
                    numeric_q_rows.append([str(c.get("column")), label,
                                           _fmt(rv), _fmt(sv)])
            for p in (c.get("proportions") or [])[:4]:
                categorical_rows.append([
                    str(c.get("column")), str(p.get("category")),
                    _pct(p.get("real_share")), _pct(p.get("synthetic_share")),
                ])

        if numeric_rows:
            story.append(Paragraph("Numerical features — central tendency and spread", styles["H3Brand"]))
            story.append(_table(numeric_rows, font_size=7.5))
            story.append(Spacer(1, 0.2 * cm))
            story.append(Paragraph("Numerical features — quantiles", styles["H3Brand"]))
            story.append(_table(numeric_q_rows, font_size=7.5))
        if len(categorical_rows) > 1:
            story.append(Paragraph("Categorical features — category proportions", styles["H3Brand"]))
            story.append(_table(categorical_rows, font_size=7.5))
    else:
        story.append(Paragraph(
            "Requirement-based project: no real dataset exists for this project, so no real-vs-synthetic "
            "statistical comparison is available. DataNova does not fabricate similarity values.",
            styles["Body"]))

    # Distribution charts (subsection of 5)
    if evalr:
        story.append(Paragraph("Distribution comparison", styles["H3Brand"]))
        cols = evalr.get("column_results") or []
        numeric = [c for c in cols if c.get("stats")]
        categorical = [c for c in cols if c.get("proportions")]
        # most informative first: worst similarity gets charted
        key = lambda c: (c.get("similarity") if c.get("similarity") is not None else 1)
        charts: List[Image] = []
        for c in sorted(numeric, key=key)[:3]:
            img = _quantile_chart(c)
            if img:
                charts.append(img)
        for c in sorted(categorical, key=key)[:3]:
            img = _category_chart(c)
            if img:
                charts.append(img)
        if charts:
            for i in range(0, len(charts), 2):
                pair = charts[i:i + 2]
                row = Table([[pair[0], pair[1] if len(pair) > 1 else ""]], colWidths=[_CONTENT_W / 2] * 2)
                row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
                story.append(row)
                story.append(Spacer(1, 0.2 * cm))
            story.append(Paragraph(
                "Charts are rendered from the stored evaluation statistics (quantiles and category proportions), "
                "not from raw data recalculation.", styles["Caption"]))
        else:
            story.append(Paragraph("Not enough per-column detail was stored to draw distribution charts.",
                                   styles["Body"]))

        # Correlation comparison (subsection of 5)
        story.append(Paragraph("Correlation / association structure", styles["H3Brand"]))
        corr = evalr.get("correlation_comparison") or {}
        if corr.get("mean_abs_difference") is not None:
            story.append(Paragraph(
                f"Average absolute difference between real and synthetic association values: "
                f"<b>{_fmt(corr['mean_abs_difference'])}</b> (max {_fmt(corr.get('max_abs_difference'))}). "
                "Lower means the relationships between columns were preserved better.", styles["Body"]))
        rm = _matrix_heatmap(corr.get("real_matrix"), corr.get("columns") or [], "Real — absolute associations")
        sm_ = _matrix_heatmap(corr.get("synthetic_matrix"), corr.get("columns") or [], "Synthetic — absolute associations")
        if rm and sm_:
            row = Table([[rm, sm_]], colWidths=[_CONTENT_W / 2] * 2)
            row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
            story.append(row)
        elif rm:
            story.append(rm)

        # Target comparison (subsection of 5)
        story.append(Paragraph("Target comparison", styles["H3Brand"]))
        if tgt.get("kind") == "classification":
            rows = [["Class", "Real share", "Synthetic share"]]
            for p in (tgt.get("class_proportions") or [])[:10]:
                rows.append([str(p.get("category")), _pct(p.get("real_share")), _pct(p.get("synthetic_share"))])
            if len(rows) > 1:
                story.append(_table(rows))
            sim = tgt.get("similarity")
            if sim is not None:
                story.append(Paragraph(f"Target similarity (TVD-based): <b>{_fmt(sim)}</b>.", styles["Small"]))
        elif tgt.get("kind") == "regression":
            st = tgt.get("stats")
            if st:
                story.append(_table([
                    ["Target statistic", "Real", "Synthetic"],
                    ["Mean", _fmt(st["mean"]["real"]), _fmt(st["mean"]["synthetic"])],
                    ["Std", _fmt(st["std"]["real"]), _fmt(st["std"]["synthetic"])],
                ]))
            sim = tgt.get("similarity")
            if sim is not None:
                story.append(Paragraph(f"Target KS-based similarity: <b>{_fmt(sim)}</b>.", styles["Small"]))
        else:
            story.append(Paragraph(str(tgt.get("reason") or "No target comparison available."), styles["Body"]))

    # 6. ML utility test (only when the experiment was run)
    if mlr:
        story.append(Paragraph("6. ML utility test", styles["H2Brand"]))
        story.append(Paragraph(mlr.get("interpretation_note", ""), styles["Small"]))
        story.append(Spacer(1, 0.2 * cm))
        split = mlr.get("split") or {}
        story.append(_info_table([
            ["Experiment A (baseline)", f"Train on {split.get('real_train_rows', '—')} real rows → test on "
                                        f"{split.get('real_test_rows', '—')} real hold-out rows"],
            ["Experiment B (synthetic)", f"Train on {split.get('synthetic_train_rows', '—')} synthetic rows → test on "
                                         f"the same {split.get('real_test_rows', '—')} real hold-out rows"],
            ["Test set", "Identical real hold-out rows for both experiments"],
        ]))
        story.append(Spacer(1, 0.25 * cm))
        a = (mlr.get("experiment_a") or {}).get("metrics") or {}
        b = (mlr.get("experiment_b") or {}).get("metrics") or {}
        is_clf = mlr.get("task") == "classification"
        metric_names = (["accuracy", "precision", "recall", "f1"] if is_clf
                        else ["mae", "mse", "rmse", "r2"])
        rows = [["Metric", "A: real-trained", "B: synthetic-trained"]]
        for m in metric_names:
            rows.append([m.upper(), _fmt(a.get(m)), _fmt(b.get(m))])
        for k, v in (mlr.get("retention") or {}).items():
            rows.append([f"Retention — {k}", _pct(v), _pct(v)])
        story.append(_table(rows))
        cm_ = a.get("confusion_matrix")
        if is_clf and cm_ and cm_.get("matrix"):
            story.append(Spacer(1, 0.2 * cm))
            story.append(Paragraph("Confusion matrix — Experiment A (rows = actual, cols = predicted)",
                                   styles["H3Brand"]))
            story.append(_table([[""] + [str(l) for l in cm_["labels"]]] +
                                [[str(l)] + [str(v) for v in row] for l, row in zip(cm_["labels"], cm_["matrix"])]))

        # Model information (subsection of 6)
        story.append(Paragraph("Model information", styles["H3Brand"]))
        model = mlr.get("model") or {}
        skipped = mlr.get("skipped_columns") or []
        story.append(_info_table([
            ["Recommended baseline model", str(model.get("label") or "—")],
            ["Why selected", str((expl.get("ml_utility") or [{}])[0].get("text") if expl.get("ml_utility")
                                 else "Chosen by DataNova's rule-based recommender from the dataset characteristics.")],
            ["Preprocessing", "Features shared by both datasets; skipped columns are reported, never imputed"],
            ["Skipped columns", ", ".join(skipped) if skipped else "none"],
            ["Split parameters", f"test_size={split.get('test_size', '—')}, random_state={split.get('random_state', '—')}"],
        ]))
        story.append(Paragraph(
            "This is a recommended baseline model, not the “best” model. Results can differ with other "
            "models or preprocessing.", styles["Caption"]))

    # ---------------------------------------------------------------- page 3+
    story.append(PageBreak())

    # 7. DataNova Score
    story.append(Paragraph("7. DataNova Score", styles["H2Brand"]))
    score_info = evalr.get("data_nova_score") or {}
    if score_info.get("value") is not None:
        story.append(Paragraph(
            f"<b>Score: {_fmt(score_info['value'])}</b> — DataNova Score, a project-defined evaluation metric.",
            styles["Body"]))
        story.append(Spacer(1, 0.2 * cm))
        comps = score_info.get("components") or {}
        weights = score_info.get("weights") or {"statistical": 0.30, "distribution": 0.30,
                                                "correlation": 0.25, "target": 0.15}
        rows = [["Component", "Similarity (0–1)", "Weight"]]
        labels = {"statistical": "Statistical (per-column KS / TVD)",
                  "distribution": "Distribution shapes",
                  "correlation": "Correlation / association structure",
                  "target": "Target distribution"}
        for k, lbl in labels.items():
            rows.append([lbl, _fmt(comps.get(k)), _pct(weights.get(k))])
        story.append(_table(rows))
        story.append(Paragraph(
            "Formula: score = 0.30·statistical + 0.30·distribution + 0.25·correlation + 0.15·target. "
            "This is DataNova's own transparent metric for this project — it is NOT an industry-standard score, "
            "and different weights would give a different number.", styles["Small"]))
    else:
        story.append(Paragraph(
            "No DataNova Score is available for this project" +
            (" (no real-vs-synthetic comparison exists for requirement-based projects)."
             if project.get("mode") == "requirements" else "."),
            styles["Body"]))

    # 8. Interpretation & methodology — stored explainability findings + computed sentences
    story.append(Paragraph("8. Interpretation & methodology", styles["H2Brand"]))
    stored_findings = (expl.get("evaluation") or []) + (expl.get("ml_utility") or [])
    computed = _target_interpretations(tgt, evalr) + _column_interpretations(evalr) + \
        (_ml_interpretations(mlr) if mlr else [])
    all_lines = computed + [f for f in stored_findings]
    if all_lines:
        for f in all_lines[:16]:
            if isinstance(f, dict):
                story.append(Paragraph(f"<b>{f.get('title', 'Note')}.</b> {f.get('text', '')}", styles["Body"]))
            else:
                story.append(Paragraph(f"• {f}", styles["Body"]))
    else:
        story.append(Paragraph("Not enough stored results to derive interpretations.", styles["Body"]))

    # 9. Limitations
    story.append(Paragraph("9. Limitations", styles["H2Brand"]))
    for line in [
        "Synthetic data does not automatically guarantee privacy, anonymity or freedom from bias.",
        "Statistical similarity does not guarantee privacy — memorised patterns from the real data can still leak.",
        "Statistical similarity does not guarantee ML usefulness, and useful data may not be statistically similar.",
        "Results depend on the dataset, generation method, preprocessing, model and evaluation configuration.",
        "This report evaluates this particular generated dataset only; conclusions may not transfer to other runs.",
        "The ML utility experiment measures downstream model performance only; privacy testing is a separate task.",
        "The DataNova Score is a project-defined weighted metric, not an industry standard.",
    ]:
        story.append(Paragraph(f"• {line}", styles["Body"]))

    # 10. Messy data configuration (only when a messy dataset was generated)
    messy_file_name = project.get("messy_file")
    messy_info = project.get("messy") or {}
    files_num = 10
    if messy_file_name:
        files_num = 11
        story.append(Paragraph("10. Messy data configuration", styles["H2Brand"]))
        counts = messy_info.get("counts") or {}
        req = messy_info.get("requested") or {}
        story.append(_info_table([
            ["Requested messiness",
             ", ".join(f"{k.replace('_', ' ')} {v}%" for k, v in req.items() if v) or "—"],
            ["Missing cells introduced", str(counts.get("missing_values", messy_info.get("missing_cells_actual", 0)))],
            ["Duplicate rows added", str(counts.get("duplicate_rows", messy_info.get("duplicate_rows_added", 0)))],
            ["Whitespace inconsistencies", str(counts.get("whitespace", 0))],
            ["Capitalization inconsistencies", str(counts.get("capitalization", 0))],
            ["Format inconsistencies", str(counts.get("format_inconsistencies", 0))],
            ["Invalid/noisy values", str(counts.get("invalid_values", 0))],
            ["Outliers introduced", str(counts.get("outliers", 0))],
            ["Messy dataset file", str(messy_file_name)],
        ]))
        story.append(Paragraph(
            "The messy dataset is a separate copy — the clean synthetic dataset above is unchanged.",
            styles["Caption"]))

    # 11 (or 10). Files & exports
    story.append(Paragraph(f"{files_num}. Files & exports", styles["H2Brand"]))
    synth_file_name = synth_preview.get("file") if isinstance(synth_preview, dict) else None
    story.append(_info_table([
        ["Synthetic dataset (CSV)", str(synth_file_name or "not saved with this project")],
        ["Messy dataset (CSV)", str(messy_file_name) + " (separate copy; clean data unchanged)" if messy_file_name else "not generated"],
        ["This report (PDF)", Path(out_path).name],
        ["DataNova Score disclaimer", "Project-defined weighted metric — not an industry standard."],
    ]))

    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph(
        f"DataNova v{APP_VERSION} · Synthetic Data Studio · "
        f"report generated from stored project results · "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        styles["Small"]))

    # ---------------------------------------------------------------- build
    def _header_footer(canvas, doc):
        canvas.saveState()
        # header band
        canvas.setFillColor(_DARK)
        canvas.rect(0, PAGE_H - 1.1 * cm, PAGE_W, 1.1 * cm, stroke=0, fill=1)
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 8)
        canvas.drawString(_MARGIN, PAGE_H - 0.72 * cm, "DATANOVA — Synthetic Data Evaluation Report")
        if LOGO_MARK_PATH.exists():
            canvas.drawImage(str(LOGO_MARK_PATH), PAGE_W - _MARGIN - 0.85 * cm, PAGE_H - 0.95 * cm,
                             width=0.85 * cm, height=0.85 * cm, preserveAspectRatio=True, mask="auto")
        # footer
        canvas.setFillColor(_GREY)
        canvas.setFont("Helvetica", 7.5)
        project_name = str(project.get("name", ""))[:60]
        canvas.drawString(_MARGIN, 0.8 * cm, f"{project_name}")
        canvas.drawRightString(PAGE_W - _MARGIN, 0.8 * cm, f"Page {doc.page}")
        canvas.setStrokeColor(_MUTED)
        canvas.setLineWidth(0.6)
        canvas.line(_MARGIN, 1.05 * cm, PAGE_W - _MARGIN, 1.05 * cm)
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=_MARGIN, rightMargin=_MARGIN, topMargin=2.1 * cm, bottomMargin=1.6 * cm,
        title=f"DataNova Report — {project.get('name', 'project')}",
    )
    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    return out_path

"""
DataNova API routes.

All endpoints return plain JSON (except downloads). The workflow state is kept
on the client: each step takes explicit inputs, so the API is stateless except
for file storage and the database. This keeps the frontend in control of the
guided workflow (Configure -> Generate -> Preview -> Evaluate -> Export).
"""
from __future__ import annotations

import random
import re
import threading
import time

import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from . import config, database
from . import schemas
from .services import (
    dataset_analyzer,
    evaluation_engine,
    explainability_engine,
    generation_validator,
    ml_utility_engine,
    messy_data,
    model_recommender,
    report_generator,
    requirement_parser,
    sdv_generator,
    semantic_generators,
    semantic_types,
    spec_synthesizer,
    uniqueness,
)

router = APIRouter(prefix="/api")

# In-memory store of uploaded DataFrames for the current session, keyed by
# upload id. Uploads are also persisted to disk under data/uploads so a server
# restart does not break a workflow.
_UPLOADS: Dict[str, pd.DataFrame] = {}
_UPLOAD_META: Dict[str, Dict[str, Any]] = {}

# Live generation progress per upload id, polled by the frontend via
# GET /upload/{id}/generation-status while POST /generate is running.
_GEN_STATUS: Dict[str, Dict[str, Any]] = {}

_UPLOAD_LOCKS: Dict[str, threading.Lock] = {}

AUTHOR = "Nuha Khan"


def _upload_lock(upload_id: str) -> threading.Lock:
    """One lock per upload so concurrent generates for the same data serialize."""
    lock = _UPLOAD_LOCKS.get(upload_id)
    if lock is None:
        lock = threading.Lock()
        _UPLOAD_LOCKS[upload_id] = lock
    return lock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_upload(upload_id: str) -> pd.DataFrame:
    df = _UPLOADS.get(upload_id)
    if df is not None:
        return df
    # Fallback: reload a persisted upload after a server restart.
    path = config.DATA_DIR / "uploads" / f"{upload_id}.csv"
    if upload_id.startswith("up_") and path.exists():
        try:
            df = pd.read_csv(path)
            _UPLOADS[upload_id] = df
            return df
        except Exception:
            pass
    raise HTTPException(status_code=404, detail="Upload not found or expired. Please upload the dataset again.")


def _save_upload(upload_id: str, df: pd.DataFrame, filename: str) -> None:
    _UPLOADS[upload_id] = df
    _UPLOAD_META[upload_id] = {"filename": filename, "uploaded_at": pd.Timestamp.now().isoformat()}
    # Persist to disk so uploads survive a server restart.
    try:
        config.ensure_dirs()
        (config.DATA_DIR / "uploads").mkdir(exist_ok=True)
        df.to_csv(config.DATA_DIR / "uploads" / f"{upload_id}.csv", index=False)
        _save_names(upload_id, filename)
    except Exception:
        pass  # persistence is best-effort; the session still works in memory


def _names_path() -> Path:
    return config.DATA_DIR / "uploads" / "_names.json"


def _load_names() -> Dict[str, str]:
    try:
        return json.loads(_names_path().read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_names(upload_id: str, filename: str) -> None:
    try:
        names = _load_names()
        names[upload_id] = filename
        _names_path().write_text(json.dumps(names, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass


def _slugify(text: str, fallback: str = "dataset") -> str:
    """Filename-safe slug for human-readable download names."""
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text[:60] or fallback


def _project_file_base(project: Dict[str, Any]) -> str:
    """Human-readable base name for a project's download files."""
    created = str(project.get("created_at") or "")
    day = created[:10].replace("-", "") or pd.Timestamp.now().strftime("%Y%m%d")
    return f"{_slugify(str(project.get('name') or 'project'))}-{day}"


def _store_synth(upload_id: str, synth: pd.DataFrame) -> None:
    """Keep the latest synthetic frame for evaluation/download without disk I/O."""
    _SYNTHS[upload_id] = synth


_SYNTHS: Dict[str, pd.DataFrame] = {}


def _get_synth(upload_id: str) -> pd.DataFrame:
    synth = _SYNTHS.get(upload_id)
    if synth is None:
        raise HTTPException(status_code=404, detail="No generated dataset found for this session. Generate again.")
    return synth


def _df_to_csv_response(df: pd.DataFrame, filename: str) -> StreamingResponse:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _jsonify_frame(df: pd.DataFrame, n: int = 20) -> List[Dict[str, Any]]:
    """First n rows as JSON-safe dicts (dates -> strings, NaN -> None)."""
    head = df.head(n).copy()
    for col in head.columns:
        if pd.api.types.is_datetime64_any_dtype(head[col]):
            head[col] = head[col].dt.strftime("%Y-%m-%d")
    return json.loads(head.to_json(orient="records", date_format="iso"))


def _sanitize_json(value: Any) -> Any:
    """Recursively replace NaN/inf floats with None so FastAPI can serialise."""
    if isinstance(value, dict):
        return {k: _sanitize_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        f = float(value)
        return None if (f != f or f in (float("inf"), float("-inf"))) else f
    return value


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@router.get("/health")
def health():
    return {"status": "ok", "app": "DataNova", "version": "1.0.0"}


# ---------------------------------------------------------------------------
# Mode A — upload + analysis + detection
# ---------------------------------------------------------------------------
@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a CSV, validate it, run the health check and target detection."""
    raw = await file.read()
    try:
        df = dataset_analyzer.load_csv_from_bytes(raw, file.filename or "upload.csv")
    except dataset_analyzer.DatasetError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    upload_id = f"up_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S%f')}"
    _save_upload(upload_id, df, file.filename)

    analysis = dataset_analyzer.analyze_dataframe(df)
    detection = _detect(df)
    schema = semantic_types.detect_schema(df)          # technical + semantic types
    preview = _jsonify_frame(df, 10)

    return {
        "upload_id": upload_id,
        "filename": file.filename,
        "preview": preview,
        "analysis": analysis,
        "detection": detection,
        "schema": schema,
    }


def _detect(df: pd.DataFrame, override: Optional[str] = None) -> Dict[str, Any]:
    """Thin wrapper so routes never import the detector directly."""
    from .services.target_detector import detect_target_and_task
    return detect_target_and_task(df, override)


@router.post("/upload/{upload_id}/detect")
def redetect_target(upload_id: str, body: schemas.DetectRequest):
    """Re-run (or override) target & task detection for an uploaded dataset."""
    df = _get_upload(upload_id)
    override = body.target if body.target else None
    return _detect(df, override)


# ---------------------------------------------------------------------------
# Mode A — SDV generation
# ---------------------------------------------------------------------------
@router.post("/upload/{upload_id}/preview")
def generation_preview(upload_id: str, body: schemas.GenerateRequest):
    """
    Generation preview: what WILL happen, without the cost of generation.
    Synthesizer choice, row counts, target/task, warnings. No synthetic rows
    are produced here.
    """
    df = _get_upload(upload_id)
    approach = sdv_generator.choose_synthesizer(
        df.drop(columns=body.exclude_columns or [], errors="ignore"),
        override=body.synthesizer,
    )
    analysis = dataset_analyzer.analyze_dataframe(df)
    return {
        "upload_id": upload_id,
        "rows_requested": body.n_rows,
        "target": body.target,
        "task": body.task,
        "approach": approach,
        "n_features": df.shape[1] - len(body.exclude_columns or []),
        "excluded_columns": body.exclude_columns or [],
        "health_warnings": analysis.get("warnings", []),
        "sample_preview": _jsonify_frame(df, 5),
        "notes": [
            "Full generation fits the synthesizer on your real data and samples new rows.",
            "Generation time depends on dataset size and the chosen synthesizer (CTGAN is the slowest).",
        ],
    }


def _postprocess_semantics(df: pd.DataFrame, synthetic: pd.DataFrame,
                           schema: Dict[str, Dict[str, Any]],
                           rng: "random.Random") -> Tuple[pd.DataFrame, List[str], int]:
    """
    Re-generate semantic-typed columns on top of the SDV output.

    SDV preserves the learned joint distribution, but its sampling of rare
    string formats (names, phones, pincodes, emails) is statistically-driven,
    not format-driven. For every column whose detected semantic type has a
    dedicated generator, we rebuild the column value-by-value so each cell is
    a valid, plausible value of that semantic type — then keep SDV's values
    for genuinely numeric/categorical columns untouched.

    Returns (frame, notes, n_regenerated_columns).
    """
    notes: List[str] = []
    rebuilt = 0
    out = synthetic.copy()
    # Only format-critical semantic types are rebuilt: SDV already preserves
    # numeric distributions and category pools well, but it cannot guarantee
    # valid NAME/PHONE/PIN/EMAIL formats or Indian place coherence.
    # Only strict-FORMAT semantic types are rebuilt with dedicated generators.
    # Category-pool columns (gender, city, state, country, names, products...)
    # are left as SDV sampled them: SDV learned the REAL dataset's own value
    # pools and their joint distribution, which is more faithful than any
    # generic generator vocabulary (and preserves city-state coherence).
    _REBUILDABLE = {"indian_phone", "email", "indian_pincode", "customer_id",
                    "generic_id", "date_of_birth"}
    for col, info in schema.items():
        if col not in out.columns:
            continue
        sem = info.get("semantic_type")
        conf = float(info.get("semantic_confidence") or 0)
        if sem not in _REBUILDABLE or conf < 0.6:
            continue
        # Numeric identifier-like columns (phones/pincodes/ids stored as
        # integers) get semantic regeneration too: copula models fit them
        # poorly (magnitudes drift wildly). The generated values are coerced
        # back to the original dtype so the column type is preserved.
        numeric_source = pd.api.types.is_numeric_dtype(df[col])
        gen = semantic_generators.generator_for(sem)
        ctx: Dict[str, Any] = {}
        if sem == "email":
            name_col = next((c for c, i2 in schema.items()
                             if i2.get("semantic_type") == "person_name" and c in out.columns), None)
            if name_col:
                ctx["name_series"] = out[name_col].astype(str).tolist()
        if sem == "indian_pincode":
            state_col = next((c for c, i2 in schema.items()
                              if i2.get("semantic_type") == "state" and c in out.columns), None)
            if state_col:
                ctx["state_series"] = out[state_col].astype(str).tolist()
        out[col] = gen.generate(len(out), rng, ctx)
        if numeric_source and sem != "date_of_birth":
            digits = out[col].astype(str).str.replace(r"\D", "", regex=True)
            out[col] = pd.to_numeric(digits, errors="coerce")
        notes.append(f"Column '{col}' regenerated with the {sem} semantic generator to guarantee valid formats.")
        rebuilt += 1
    return out, notes, rebuilt


@router.get("/upload/{upload_id}/generation-status")
def generation_status(upload_id: str):
    """Poll live generation progress (stages: analyzing→fitting→sampling→validating→complete)."""
    return _GEN_STATUS.get(upload_id, {"stage": "idle"})


@router.post("/upload/{upload_id}/generate")
def generate(upload_id: str, body: schemas.GenerateRequest):
    """Fit the synthesizer ONCE, sample, re-apply semantic formats, enforce uniqueness."""
    import random as _random

    df = _get_upload(upload_id)
    if body.n_rows < 1:
        raise HTTPException(status_code=400, detail="n_rows must be at least 1.")
    if body.n_rows > config.MAX_ROWS_FULL_GENERATION:
        raise HTTPException(status_code=400,
                            detail=f"n_rows is capped at {config.MAX_ROWS_FULL_GENERATION:,} for resource safety.")

    lock = _upload_lock(upload_id)
    if not lock.acquire(blocking=False):
        raise HTTPException(status_code=409,
                            detail="A generation is already running for this dataset. Wait for it to finish.")

    def _set_stage(stage: str, **extra: Any) -> None:
        _GEN_STATUS[upload_id] = {"stage": stage, **extra}

    try:
        _set_stage("analyzing")
        schema = semantic_types.detect_schema(df)
        rng = _random.Random(42)

        _set_stage("fitting")
        fitted = sdv_generator.fit_synthesizer(
            df,
            exclude_columns=body.exclude_columns,
            synthesizer_override=body.synthesizer,
            advanced=body.advanced,
        )

        def _sample(batch: int) -> pd.DataFrame:
            return fitted.sample(batch)

        _set_stage("sampling")
        t_sample0 = time.time()
        synth = _sample(body.n_rows)
        sample_seconds = round(time.time() - t_sample0, 1)
        sampled_rows = int(len(synth))

        # --- semantic post-processing (format validity for names/phones/...) ---
        synth, sem_notes, n_rebuilt = _postprocess_semantics(df, synth, schema, rng)

        # --- uniqueness enforcement (dedupe + top-up sampling from the SAME
        #     fitted model — no refitting, so this stays fast) ---
        _set_stage("validating", rows_generated=sampled_rows)

        def make_more(k: int) -> pd.DataFrame:
            extra = _sample(k)
            extra, _, _2b = _postprocess_semantics(df, extra, schema, rng)
            return extra

        synth, uni_report = uniqueness.enforce_unique_rows(
            synth, original=df, n_requested=body.n_rows,
            make_more=make_more,
            unique_columns=[],
        )

        # --- source-fidelity gate: did generation actually follow the upload? ---
        fidelity = sdv_generator.structure_violations(synth, df)
        retried = False
        if (fidelity["invented_category_values"] > 0 or fidelity["out_of_range_numeric"] > 0) \
                and not body.synthesizer and fitted.approach != "GaussianCopula":
            # Auto-retry ONCE with the structure-preserving fallback (copula).
            # Only when the user left the synthesizer on Auto.
            _set_stage("fitting", fallback=True)
            fitted = sdv_generator.fit_synthesizer(
                df, exclude_columns=body.exclude_columns,
                synthesizer_override="GaussianCopula", advanced=body.advanced,
            )
            _set_stage("sampling")
            t_sample0 = time.time()
            synth = fitted.sample(body.n_rows)
            sample_seconds = round(time.time() - t_sample0, 1)
            synth, sem_notes, n_rebuilt = _postprocess_semantics(df, synth, schema, rng)

            def make_more(k: int) -> pd.DataFrame:
                extra = fitted.sample(k)
                extra, _, _2b = _postprocess_semantics(df, extra, schema, rng)
                return extra

            synth, uni_report = uniqueness.enforce_unique_rows(
                synth, original=df, n_requested=body.n_rows,
                make_more=make_more, unique_columns=[],
            )
            fidelity = sdv_generator.structure_violations(synth, df)
            retried = True
    except RuntimeError as exc:
        _set_stage("error", detail=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))
    except HTTPException:
        _set_stage("error", detail="Generation failed.")
        raise
    finally:
        lock.release()

    _set_stage("complete", rows_generated=int(len(synth)))

    # --- generation quality check ---
    column_specs = [
        {
            "name": col,
            "technical_type": schema.get(col, {}).get("technical_type"),
            "semantic_type": schema.get(col, {}).get("semantic_type"),
        }
        for col in df.columns if col not in (body.exclude_columns or [])
    ]
    for col in (body.exclude_columns or []):
        if col in df.columns:
            column_specs.append({"name": col, "technical_type": "string", "semantic_type": "generic_id"})
    quality = generation_validator.validate_generation(
        synth, column_specs=column_specs, n_requested=body.n_rows,
        unique_columns=[], original=df,
    )

    _store_synth(upload_id, synth)

    validation_warnings = list(sdv_generator.validate_synthetics(synth, df))
    validation_warnings.extend([n for n in uni_report.get("notes", [])])
    if n_rebuilt:
        validation_warnings.append(
            f"{n_rebuilt} semantic column(s) were re-generated with dedicated format-aware generators "
            "(SDV learned their distribution; the generator guarantees valid formats)."
        )
    if retried:
        validation_warnings.append(
            "First generation attempt did not follow the source data closely enough, so DataNova "
            "automatically regenerated with GaussianCopula (structure-preserving fallback)."
        )
    if fidelity["details"]:
        validation_warnings.append(
            "Remaining source-fidelity warnings: " + "; ".join(fidelity["details"][:5]) +
            ". You can regenerate or choose a different synthesizer under generation options."
        )

    return {
        "upload_id": upload_id,
        "approach": fitted.approach,
        "reason": fitted.reason,
        "fit_seconds": fitted.fit_seconds,
        "sample_seconds": sample_seconds,
        "n_rows_generated": int(len(synth)),
        "n_rows_requested": body.n_rows,
        "validation_warnings": validation_warnings,
        "quality_check": quality,
        "uniqueness": uni_report,
        "fidelity": fidelity,
        "schema": schema,
        "preview": _jsonify_frame(synth, config.MAX_PREVIEW_ROWS),
        "columns": list(synth.columns),
    }


@router.get("/upload/{upload_id}/synthetic-preview")
def synthetic_preview(upload_id: str, n: int = 20):
    """Preview the stored synthetic dataset."""
    synth = _get_synth(upload_id)
    return {"rows": _jsonify_frame(synth, n), "total_rows": int(len(synth))}


# ---------------------------------------------------------------------------
# Session-level messy data (Mode 1: applied AFTER generation, original kept)
# ---------------------------------------------------------------------------
@router.post("/upload/{upload_id}/messy")
def session_messy(upload_id: str, body: schemas.MessyConfigRequest):
    """Apply user-specified messiness to the generated synthetic dataset.

    Produces a SEPARATE messy frame; the clean generation stays available for
    download/evaluation. Actual percentages are reported, not just requested.
    """
    synth = _get_synth(upload_id)

    def pct(v: float) -> float:
        return max(0.0, min(100.0, float(v or 0)))

    cfg = {
        "missing_values": pct(body.missing_values),
        "duplicate_rows": pct(body.duplicate_rows),
        "whitespace": pct(body.whitespace),
        "capitalization": pct(body.capitalization),
        "format_inconsistency": pct(body.format_inconsistency),
        "invalid_values": pct(body.invalid_values),
        "outliers": pct(body.outliers),
    }
    if all(v == 0 for v in cfg.values()):
        raise HTTPException(status_code=400, detail="Set at least one messiness percentage above 0.")
    if sum(cfg.values()) > 60:
        raise HTTPException(status_code=400,
                            detail="Total messiness above 60% would destroy the dataset; keep the combined total at or below 60%.")

    result = messy_data.make_messy(synth, {k: v / 100.0 for k, v in cfg.items()}, seed=42)
    messy_df = result["dataframe"]
    summary = result["summary"]

    # Keep BOTH frames; the messy copy is served under a distinct key.
    _SYNTHS[f"{upload_id}__messy"] = messy_df
    _UPLOAD_META[f"{upload_id}__messy"] = {
        "filename": f"{_UPLOAD_META.get(upload_id, {}).get('filename', 'dataset')} messy copy",
        "uploaded_at": pd.Timestamp.now().isoformat(),
    }
    return {
        "summary": summary,
        "actual_percentages": {
            "missing_values": round(summary["counts"]["missing_values"]
                                    / max(summary["total_cells"], 1) * 100, 2),
            "duplicate_rows": round(summary["duplicate_rows_added"]
                                    / max(summary["original_rows"], 1) * 100, 2),
        },
        "preview": _jsonify_frame(messy_df, config.MAX_PREVIEW_ROWS),
    }


@router.get("/upload/{upload_id}/messy/download")
def session_messy_download(upload_id: str):
    """Download the messy COPY of the session's synthetic dataset."""
    key = f"{upload_id}__messy"
    if key not in _SYNTHS:
        raise HTTPException(status_code=404, detail="No messy dataset generated for this session yet.")
    meta = _UPLOAD_META.get(key, {})
    source_name = str(meta.get("filename") or "dataset").replace(" messy copy", "")
    stem = re.sub(r"\.csv$", "", source_name, flags=re.IGNORECASE)
    base = _slugify(stem, "dataset")
    stamp = pd.Timestamp.now().strftime("%Y%m%d-%H%M")
    return _df_to_csv_response(_SYNTHS[key], f"{base}-synthetic-messy-{stamp}.csv")


# ---------------------------------------------------------------------------
# Evaluation + ML utility (work for both modes via saved datasets)
# ---------------------------------------------------------------------------

def _load_pair(body: schemas.EvaluateRequest):
    """Load (real, synthetic) frames for evaluation from either mode."""
    if body.upload_id:
        real = _get_upload(body.upload_id)
        synth = _get_synth(body.upload_id)
        return real, synth
    if body.project_id:
        proj = database.get_project(body.project_id)
        if not proj:
            raise HTTPException(status_code=404, detail="Project not found.")
        real = _load_stored_real(proj)
        synth = _load_stored_synth(proj)
        return real, synth
    raise HTTPException(status_code=400, detail="Provide upload_id or project_id.")


def _load_stored_real(proj: Dict[str, Any]) -> pd.DataFrame:
    """Real frame of a saved project (uploaded CSV or requirement-mode source)."""
    if proj.get("upload_ref"):
        path = config.DATA_DIR / "uploads" / f"{proj['upload_ref']}.csv"
        if path.exists():
            return pd.read_csv(path)
    if proj.get("dataset_filename"):
        # Legacy projects persisted the source filename directly.
        path = config.DATA_DIR / "uploads" / str(proj["dataset_filename"])
        if path.exists():
            return pd.read_csv(path)
    # Requirement mode: no real dataset exists; evaluation compares nothing.
    raise HTTPException(status_code=404, detail="This project has no real dataset to compare against.")


def _load_stored_synth(proj: Dict[str, Any]) -> pd.DataFrame:
    """Synthetic frame of a saved project, from disk cache."""
    synth_name = (proj.get("synth_preview") or {}).get("file") if isinstance(proj.get("synth_preview"), dict) else None
    if synth_name:
        path = config.DATA_DIR / "synthetic" / synth_name
        if path.exists():
            return pd.read_csv(path)
    raise HTTPException(status_code=404, detail="Stored synthetic dataset file is missing. Regenerate the data.")


@router.post("/evaluate")
def evaluate(body: schemas.EvaluateRequest):
    """Full statistical evaluation of real vs synthetic."""
    real, synth = _load_pair(body)
    try:
        result = evaluation_engine.evaluate(real, synth, target=body.target)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}")
    return _sanitize_json(result)


@router.post("/ml-utility/recommendation")
def ml_recommendation(body: schemas.MLRequest):
    """Explainable baseline model recommendation for the detected task."""
    real = _get_upload(body.upload_id) if body.upload_id else None
    if real is None:
        raise HTTPException(status_code=400, detail="upload_id is required for model recommendation.")
    rec = model_recommender.recommend_model(real, body.task, body.target)
    return rec


@router.post("/ml-utility/run")
def ml_run(body: schemas.MLRunRequest):
    """Run the two-experiment ML utility test."""
    real, synth = _load_pair(schemas.EvaluateRequest(upload_id=body.upload_id, project_id=body.project_id))
    try:
        result = ml_utility_engine.run_ml_utility(real, synth, target=body.target, task=body.task, model_key=body.model_key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ML utility test failed: {exc}")
    return _sanitize_json(result)


@router.post("/explain")
def explain(body: schemas.ExplainRequest):
    """Generate explanations from stored evaluation / ML results."""
    findings = {
        "evaluation": explainability_engine.explain_evaluation(body.evaluation or {}),
        "ml_utility": explainability_engine.explain_ml_utility(body.ml_utility or {}),
        "generation": explainability_engine.explain_generation(
            body.generation_approach or "SDV", body.generation_reason or "", body.warnings or []),
    }
    return findings


# ---------------------------------------------------------------------------
# Mode B — natural-language requirements
# ---------------------------------------------------------------------------
@router.post("/requirements/parse")
def parse_requirements(body: schemas.ParseRequest):
    """Tell DataNova What You Want: free text -> structured spec + clarifications."""
    if not body.text or not body.text.strip():
        raise HTTPException(status_code=400, detail="Please describe the dataset you want.")
    spec = requirement_parser.parse_requirements(body.text)
    return {"spec": requirement_parser.spec_to_dict(spec)}


@router.post("/requirements/generate")
def generate_from_requirements(body: schemas.RequirementGenerateRequest):
    """Generate a dataset from a (possibly edited) structured spec."""
    spec = requirement_parser.RequirementSpec(
        rows=body.spec.rows,
        task=body.spec.task,
        target=body.spec.target,
        features=[c.model_dump() for c in body.spec.columns],   # v2 semantic schema
        constraints=body.spec.constraints,
        class_distribution=body.spec.class_distribution,
        domain=body.spec.domain,
        raw_text=body.spec.raw_text,
    )
    spec.warnings = list(body.spec.warnings or [])
    spec.columns = [c.model_dump() for c in body.spec.columns]
    spec.unique = body.spec.unique
    spec.country = body.spec.country

    # Rows must be confirmed by the user: if the parser could not find any, the
    # spec has to have been edited before calling this endpoint.
    if not spec.rows:
        raise HTTPException(status_code=400, detail="Row count is required. Answer the clarification questions first.")
    if not spec.columns:
        raise HTTPException(status_code=400, detail="Add at least one column in the schema editor before generating.")

    try:
        result = spec_synthesizer.generate_from_spec(spec, seed=body.seed)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}")

    df = result["dataframe"]
    # Store the generated frame for this session so the UI can offer CSV download
    req_session_id = f"req_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S%f')}"
    _store_synth(req_session_id, df)
    _UPLOAD_META[req_session_id] = {
        "filename": f"{spec.target or 'synthetic'} dataset (from requirements)",
        "uploaded_at": pd.Timestamp.now().isoformat(),
    }

    # Requirement-mode datasets cannot be evaluated against real data, so we
    # only return a self-descriptive summary, never fake comparison scores.
    target_col = spec.target if (spec.target and spec.target in df.columns) else None
    if target_col:
        target_summary = (
            {"positive_share": round(float(df[target_col].mean()), 4),
             "counts": {str(k): int(v) for k, v in df[target_col].value_counts().items()}}
            if spec.task != "regression" else
            {"mean": round(float(df[target_col].mean()), 4),
             "std": round(float(df[target_col].std()), 4)}
        )
    else:
        target_summary = {"note": "No ML target was requested; the dataset contains features only."}
    summary = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "target": target_col or "",
        "task": spec.task or "classification",
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "numeric_summary": {
            c: {
                "mean": round(float(df[c].mean()), 4),
                "std": round(float(df[c].std()), 4),
                "min": round(float(df[c].min()), 4),
                "max": round(float(df[c].max()), 4),
            }
            for c in df.select_dtypes("number").columns
        },
        "target_summary": target_summary,
    }
    return {
        "spec": body.spec.model_dump(),
        "summary": summary,
        "description": result["description"],
        "warnings": result["warnings"],
        "uniqueness": result["uniqueness"],
        "quality_check": result["validation"],
        "session_id": req_session_id,
        "preview": _jsonify_frame(df, config.MAX_PREVIEW_ROWS),
    }


# ---------------------------------------------------------------------------
# Projects (history)
# ---------------------------------------------------------------------------
@router.get("/projects")
def projects_list():
    """List all saved projects (dashboard history)."""
    return {"projects": database.list_projects()}


@router.get("/projects/{project_id}")
def projects_get(project_id: int):
    proj = database.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found.")
    return proj


@router.delete("/projects/{project_id}")
def projects_delete(project_id: int):
    if not database.delete_project(project_id):
        raise HTTPException(status_code=404, detail="Project not found.")
    return {"deleted": True}


@router.post("/projects/save")
def projects_save(body: schemas.SaveProjectRequest):
    """
    Persist the current workflow as a project (called at the Export step).
    Saves the synthetic CSV to disk so the project can be reopened later.
    """
    synth_df: Optional[pd.DataFrame] = None
    if body.upload_id and body.upload_id in _SYNTHS:
        synth_df = _SYNTHS[body.upload_id]

    config.ensure_dirs()
    slug_base = f"{_slugify(body.name, 'project')}-{pd.Timestamp.now().strftime('%Y%m%d')}"

    # Persist synthetic CSV for reloads, named after the project so downloads
    # are human-readable.
    synth_filename = None
    synth_preview_rows = None
    if synth_df is not None:
        synth_dir = config.DATA_DIR / "synthetic"
        synth_dir.mkdir(exist_ok=True)
        synth_filename = f"{slug_base}-synthetic.csv"
        synth_df.to_csv(synth_dir / synth_filename, index=False)
        synth_preview_rows = _jsonify_frame(synth_df, 5)

    # Every project needs a human-readable name (§34): derive one when the
    # user leaves it blank instead of storing an empty string.
    project_name = (body.name or "").strip()
    if not project_name:
        if body.mode == "requirements":
            project_name = f"{body.target or 'Custom'} Synthetic Dataset"
        else:
            stem = re.sub(r"\.csv$", "", body.dataset_filename or "", flags=re.IGNORECASE).strip()
            project_name = f"{_slugify(stem, 'dataset').replace('-', ' ').title()} Synthetic Dataset" if stem \
                else "Untitled Synthetic Dataset"

    fields = {
        "name": project_name,
        "upload_ref": body.upload_id if body.mode == "upload" else None,
        "mode": body.mode,
        "dataset_filename": body.dataset_filename,
        "rows_real": body.rows_real,
        "columns": body.columns,
        "target": body.target,
        "task": body.task,
        "target_overridden": body.target_overridden,
        "synthesizer": body.synthesizer,
        "generation_reason": body.generation_reason,
        "rows_requested": body.rows_requested,
        "rows_synthetic": body.rows_synthetic,
        "fit_seconds": body.fit_seconds,
        "sample_seconds": body.sample_seconds,
        "analysis": json.dumps(body.analysis) if body.analysis else None,
        "detection": json.dumps(body.detection) if body.detection else None,
        "evaluation": json.dumps(body.evaluation) if body.evaluation else None,
        "ml_utility": json.dumps(body.ml_utility) if body.ml_utility else None,
        "explanation": json.dumps(body.explanation) if body.explanation else None,
        "requirement_spec": json.dumps(body.requirement_spec) if body.requirement_spec else None,
        "warnings": json.dumps(body.warnings or []),
        "synth_preview": json.dumps({"file": synth_filename, "rows": synth_preview_rows}),
    }
    project_id = database.save_project(fields)
    return {"project_id": project_id, "synthetic_file": synth_filename}


@router.post("/projects/{project_id}/report")
def generate_report(project_id: int, body: schemas.ReportRequest | None = None):
    """Generate the PDF report for a saved project."""
    proj = database.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found.")

    config.ensure_dirs()
    filename = f"{_project_file_base(proj)}-report.pdf"
    out_path = config.REPORT_DIR / filename
    try:
        report_generator.generate_report(proj, out_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {exc}")

    database.update_project(project_id, {
        "report_path": str(out_path),
        "report_generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
    })
    return {"report_path": str(out_path), "filename": filename}


@router.get("/projects/{project_id}/report/download")
def download_report(project_id: int):
    """Download the previously generated PDF report."""
    proj = database.get_project(project_id)
    if not proj or not proj.get("report_path"):
        raise HTTPException(status_code=404, detail="No report generated yet for this project.")
    path = Path(proj["report_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report file missing on disk.")
    return FileResponse(path, media_type="application/pdf", filename=path.name)


@router.patch("/projects/{project_id}")
def projects_rename(project_id: int, body: schemas.RenameRequest):
    """Rename a saved project. Future report/CSV downloads use the new name."""
    proj = database.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found.")
    new_name = (body.name or "").strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="Project name cannot be empty.")
    if len(new_name) > 120:
        raise HTTPException(status_code=400, detail="Project name is too long (max 120 characters).")
    database.update_project(project_id, {"name": new_name})
    return {"id": project_id, "name": new_name}


@router.get("/projects/{project_id}/files")
def projects_files(project_id: int):
    """Existence-aware file listing for a saved project (CSV + PDF)."""
    proj = database.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found.")

    preview = proj.get("synth_preview") or {}
    synth_file = preview.get("file") if isinstance(preview, dict) else None
    csv_ok = False
    if synth_file:
        csv_ok = (config.DATA_DIR / "synthetic" / synth_file).exists()
    report_ok = bool(proj.get("report_path")) and Path(proj["report_path"]).exists()

    return {
        "synthetic_csv": {"exists": csv_ok, "file": synth_file},
        "report_pdf": {"exists": report_ok, "generated_at": proj.get("report_generated_at")},
        "messy_csv": {"exists": bool(proj.get("messy_file")), "file": proj.get("messy_file")},
    }


# ---------------------------------------------------------------------------
# Messy data (Make Data Messy) — user-controlled, original never modified
# ---------------------------------------------------------------------------
def _project_synth_frame(project_id: int) -> tuple:
    """Load the stored synthetic CSV for a project (source for messy data)."""
    proj = database.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found.")
    preview = proj.get("synth_preview") or {}
    synth_file = preview.get("file") if isinstance(preview, dict) else None
    if not synth_file:
        raise HTTPException(status_code=404, detail="This project has no stored synthetic dataset.")
    path = config.DATA_DIR / "synthetic" / synth_file
    if not path.exists():
        raise HTTPException(status_code=404, detail="Synthetic dataset file missing on disk.")
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read the stored synthetic dataset: {exc}")
    return proj, df


@router.post("/projects/{project_id}/messy/preview")
def messy_preview(project_id: int, body: schemas.MessyConfigRequest):
    """Estimate how many cells/rows each requested messiness type would affect."""
    proj, df = _project_synth_frame(project_id)
    rows, cols = df.shape
    total = rows * cols

    def pct(v: float) -> float:
        return max(0.0, min(100.0, float(v or 0)))

    cfg = {
        "missing_values": pct(body.missing_values),
        "duplicate_rows": pct(body.duplicate_rows),
        "whitespace": pct(body.whitespace),
        "capitalization": pct(body.capitalization),
        "format_inconsistency": pct(body.format_inconsistency),
        "invalid_values": pct(body.invalid_values),
        "outliers": pct(body.outliers),
    }
    schema = semantic_types.detect_schema(df)
    textish = {c for c, s in schema.items()
               if s.get("semantic_type") in ("generic_text", "person_name", "city", "state",
                                             "country", "address", "email", "indian_phone",
                                             "category", "product_name", "company_name")}
    numericish = {c for c, s in schema.items()
                  if s.get("semantic_type") in ("numerical", "salary", "age")}
    formatable = {c for c, s in schema.items()
                  if s.get("semantic_type") in ("indian_phone", "email", "date", "datetime", "date_of_birth")}

    return {
        "dataset": {"rows": int(rows), "columns": int(cols), "total_cells": int(total),
                    "filename": (proj.get("synth_preview") or {}).get("file")},
        "requested_config": cfg,
        "estimates": {
            "missing_cells": int(round(total * cfg["missing_values"] / 100)),
            "duplicate_rows": int(round(rows * cfg["duplicate_rows"] / 100)),
            "whitespace_cells": int(round(sum(1 for c in textish if c in df.columns) * rows * cfg["whitespace"] / 100)),
            "capitalization_cells": int(round(sum(1 for c in textish if c in df.columns) * rows * cfg["capitalization"] / 100)),
            "format_cells": int(round(sum(1 for c in formatable if c in df.columns) * rows * cfg["format_inconsistency"] / 100)),
            "invalid_cells": int(round(sum(1 for c in numericish if c in df.columns) * rows * cfg["invalid_values"] / 100)),
            "outlier_cells": int(round(sum(1 for c in numericish if c in df.columns) * rows * cfg["outliers"] / 100)),
            "text_columns": sorted(textish & set(df.columns)),
            "numeric_columns": sorted(numericish & set(df.columns)),
            "formatable_columns": sorted(formatable & set(df.columns)),
        },
    }


@router.post("/projects/{project_id}/messy")
def messy_generate(project_id: int, body: schemas.MessyConfigRequest):
    """Generate the messy COPY of the stored synthetic dataset (original untouched)."""
    proj, df = _project_synth_frame(project_id)

    def pct(v: float) -> float:
        return max(0.0, min(100.0, float(v or 0)))

    cfg = {
        "missing_values": pct(body.missing_values),
        "duplicate_rows": pct(body.duplicate_rows),
        "whitespace": pct(body.whitespace),
        "capitalization": pct(body.capitalization),
        "format_inconsistency": pct(body.format_inconsistency),
        "invalid_values": pct(body.invalid_values),
        "outliers": pct(body.outliers),
    }
    if all(v == 0 for v in cfg.values()):
        raise HTTPException(status_code=400, detail="Set at least one messiness percentage above 0.")
    if sum(cfg.values()) > 60:
        raise HTTPException(status_code=400,
                            detail="Total messiness above 60% would destroy the dataset; keep the combined total at or below 60%.")
    cfg_fractions = {k: v / 100.0 for k, v in cfg.items()}

    result = messy_data.make_messy(df, cfg_fractions, seed=42)
    messy_df = result["dataframe"]
    summary = result["summary"]

    config.ensure_dirs()
    base = _project_file_base(proj)
    messy_name = f"{base}-messy.csv"
    messy_path = config.DATA_DIR / "synthetic" / messy_name
    messy_df.to_csv(messy_path, index=False)

    summary["file"] = messy_name
    database.update_project(project_id, {
        "messy_file": messy_name,
        "messy_summary": json.dumps(summary, ensure_ascii=False),
    })
    return {
        "file": messy_name,
        "summary": summary,
        "preview": _jsonify_frame(messy_df, config.MAX_PREVIEW_ROWS),
    }


@router.get("/projects/{project_id}/messy/download")
def messy_download(project_id: int):
    """Download the messy CSV copy (original clean file is never touched)."""
    proj = database.get_project(project_id)
    if not proj or not proj.get("messy_file"):
        raise HTTPException(status_code=404, detail="No messy dataset generated for this project yet.")
    path = config.DATA_DIR / "synthetic" / proj["messy_file"]
    if not path.exists():
        raise HTTPException(status_code=404, detail="Messy dataset file missing on disk.")
    return FileResponse(path, media_type="text/csv", filename=f"{_project_file_base(proj)}-messy.csv")


@router.get("/projects/{project_id}/synthetic/download")
def download_synthetic_csv(project_id: int):
    """Download the stored synthetic CSV for a saved project."""
    proj = database.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found.")
    preview = proj.get("synth_preview") or {}
    synth_file = preview.get("file") if isinstance(preview, dict) else None
    if not synth_file:
        raise HTTPException(status_code=404, detail="This project has no stored synthetic dataset.")
    path = config.DATA_DIR / "synthetic" / synth_file
    if not path.exists():
        raise HTTPException(status_code=404, detail="Synthetic dataset file missing on disk.")
    return FileResponse(path, media_type="text/csv", filename=f"{_project_file_base(proj)}-synthetic.csv")


# ---------------------------------------------------------------------------
# Direct session downloads (before the project is saved)
# ---------------------------------------------------------------------------
@router.get("/upload/{upload_id}/synthetic/download")
def download_session_csv(upload_id: str):
    synth = _get_synth(upload_id)
    meta = _UPLOAD_META.get(upload_id, {})
    source_name = str(meta.get("filename") or "dataset")
    if upload_id.startswith("req_"):
        base = _slugify(source_name, "synthetic-dataset")
    else:
        stem = re.sub(r"\.csv$", "", source_name, flags=re.IGNORECASE)
        base = f"{_slugify(stem, 'dataset')}-synthetic"
    stamp = pd.Timestamp.now().strftime("%Y%m%d-%H%M")
    return _df_to_csv_response(synth, f"{base}-{stamp}.csv")

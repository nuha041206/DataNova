"""
Full end-to-end test of the DataNova HTTP API (both modes).

Requires the backend running on 127.0.0.1:8000:
    cd datanova/backend && uvicorn app.main:app --port 8000

Run:  python scripts/e2e_api_test.py
"""
import io
import json
import sys
import warnings
from pathlib import Path

import pandas as pd
import requests

warnings.filterwarnings("ignore")

BASE = "http://127.0.0.1:8000/api"
PASS, FAIL = "PASS", "FAIL"
results = []


def step(name, fn):
    try:
        out = fn()
        results.append((PASS, name))
        return out
    except Exception as exc:
        results.append((FAIL, f"{name}: {type(exc).__name__}: {exc}"))
        raise


def main():
    csv_path = Path(__file__).resolve().parent / "sample_data" / "telco_churn.csv"

    # ---------- Mode A ----------
    with open(csv_path, "rb") as f:
        r = requests.post(f"{BASE}/upload", files={"file": ("telco_churn.csv", f, "text/csv")})
    up = step("upload + analyze + detect", lambda: _check(r, 200).json())
    upload_id = up["upload_id"]
    assert up["detection"]["suggested_target"] == "Churn"
    assert up["analysis"]["rows"] == 1200

    det = step("target override (total_charges -> rejected as target is allowed)",
               lambda: _check(requests.post(f"{BASE}/upload/{upload_id}/detect", json={"target": "TotalCharges"}), 200).json())
    assert det["task"] == "regression"  # numeric continuous target

    # put detection back to the real target
    det = _check(requests.post(f"{BASE}/upload/{upload_id}/detect", json={"target": "Churn"}), 200).json()

    preview = step("generation preview", lambda: _check(requests.post(
        f"{BASE}/upload/{upload_id}/preview",
        json={"n_rows": 300, "target": "Churn", "task": "classification", "exclude_columns": ["customerID"]},
    ), 200).json())
    assert preview["approach"]["synthesizer"] == "GaussianCopula"

    gen = step("SDV generation (300 rows)", lambda: _check(requests.post(
        f"{BASE}/upload/{upload_id}/generate",
        json={"n_rows": 300, "target": "Churn", "task": "classification", "exclude_columns": ["customerID"]},
    ), 200).json())
    assert gen["n_rows_generated"] == 300

    ev = step("evaluation", lambda: _check(requests.post(
        f"{BASE}/evaluate", json={"upload_id": upload_id, "target": "Churn"}), 200).json())
    assert 0 <= ev["data_nova_score"]["value"] <= 1

    rec = step("model recommendation", lambda: _check(requests.post(
        f"{BASE}/ml-utility/recommendation",
        json={"upload_id": upload_id, "task": "classification", "target": "Churn"}), 200).json())
    assert rec["model_key"]

    ml = step("ML utility run", lambda: _check(requests.post(
        f"{BASE}/ml-utility/run",
        json={"upload_id": upload_id, "target": "Churn", "task": "classification", "model_key": rec["model_key"]}),
        200).json())
    assert "accuracy" in ml["experiment_a"]["metrics"]

    expl = step("explainability", lambda: _check(requests.post(f"{BASE}/explain", json={
        "evaluation": ev, "ml_utility": ml,
        "generation_approach": gen["approach"], "generation_reason": gen["reason"],
        "warnings": gen["validation_warnings"],
    }), 200).json())
    assert expl["evaluation"]

    saved = step("save project", lambda: _check(requests.post(f"{BASE}/projects/save", json={
        "name": "E2E telco churn", "mode": "upload", "upload_id": upload_id, "dataset_filename": "telco_churn.csv",
        "rows_real": 1200, "columns": 8, "target": "Churn", "task": "classification",
        "synthesizer": gen["approach"], "generation_reason": gen["reason"],
        "rows_requested": 300, "rows_synthetic": gen["n_rows_generated"],
        "fit_seconds": gen["fit_seconds"], "sample_seconds": gen["sample_seconds"],
        "analysis": up["analysis"], "detection": det, "evaluation": ev, "ml_utility": ml,
        "explanation": expl, "warnings": gen["validation_warnings"],
    }), 200).json())
    pid = saved["project_id"]

    step("generate PDF report", lambda: _check(requests.post(f"{BASE}/projects/{pid}/report", json={}), 200).json())
    rep = requests.get(f"{BASE}/projects/{pid}/report/download")
    assert rep.status_code == 200 and rep.headers["content-type"] == "application/pdf", "report download"
    results.append((PASS, "PDF report download"))

    csv_resp = requests.get(f"{BASE}/projects/{pid}/synthetic/download")
    assert csv_resp.status_code == 200 and "customerID" in csv_resp.text, "project CSV download"
    results.append((PASS, "project synthetic CSV download"))

    sess_csv = requests.get(f"{BASE}/upload/{upload_id}/synthetic/download")
    assert sess_csv.status_code == 200, "session CSV download"
    results.append((PASS, "session synthetic CSV download"))

    # ---------- Mode B ----------
    parsed = step("Mode B: parse requirements", lambda: _check(requests.post(
        f"{BASE}/requirements/parse",
        json={"text": "I need a classification dataset with 5000 records for predicting customer churn. "
                      "Include age, income, tenure, monthly charges and contract type. "
                      "The target should be churn with approximately 20% positive cases."}),
        200).json())
    spec = parsed["spec"]
    assert spec["rows"] == 5000 and (spec["target"] or "").lower() == "churn"

    # ambiguous request should produce clarifications
    amb = _check(requests.post(f"{BASE}/requirements/parse", json={"text": "I want some data."}), 200).json()
    assert amb["spec"]["clarifications"], "ambiguous text must raise clarifications"
    results.append((PASS, "Mode B: clarifications for ambiguous request"))

    genb = step("Mode B: generate from spec", lambda: _check(requests.post(
        f"{BASE}/requirements/generate",
        json={"spec": {**spec, "rows": 500}, "seed": 42}), 200).json())
    assert genb["summary"]["rows"] == 500

    savedb = step("Mode B: save project", lambda: _check(requests.post(f"{BASE}/projects/save", json={
        "name": "E2E requirements churn", "mode": "requirements",
        "columns": len(genb["summary"]["columns"]), "target": genb["summary"]["target"],
        "task": genb["summary"]["task"], "rows_synthetic": 500,
        "requirement_spec": genb["spec"], "warnings": genb["warnings"],
    }), 200).json())

    step("Mode B: PDF report", lambda: _check(requests.post(f"{BASE}/projects/{savedb['project_id']}/report", json={}), 200).json())

    # ---------- History ----------
    plist = step("projects list", lambda: _check(requests.get(f"{BASE}/projects"), 200).json())
    assert len(plist["projects"]) >= 2
    detail = _check(requests.get(f"{BASE}/projects/{pid}"), 200).json()
    assert detail["evaluation"]["data_nova_score"]["value"] == ev["data_nova_score"]["value"]

    print()
    for mark, name in results:
        print(f"{mark}  {name}")
    fails = [n for m, n in results if m == FAIL]
    print(f"\n{len(results) - len(fails)}/{len(results)} E2E steps passed")


def _check(r: requests.Response, expected: int) -> requests.Response:
    if r.status_code != expected:
        raise AssertionError(f"HTTP {r.status_code}: {r.text[:200]}")
    return r


if __name__ == "__main__":
    main()

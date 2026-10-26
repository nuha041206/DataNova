"""
End-to-end smoke test of DataNova backend services (no HTTP layer).

Exercises: analyzer -> detection -> SDV generation -> evaluation ->
ML utility -> explanation -> PDF report -> SQLite persistence.

Run:  python scripts/smoke_test.py
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from app.services import (
    dataset_analyzer,
    evaluation_engine,
    explainability_engine,
    ml_utility_engine,
    model_recommender,
    report_generator,
    requirement_parser,
    sdv_generator,
    spec_synthesizer,
)
from app import database

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
    sample = Path(__file__).resolve().parent / "sample_data" / "telco_churn.csv"
    if not sample.exists():
        from make_sample_data import telco_churn
        sample.parent.mkdir(exist_ok=True)
        telco_churn().to_csv(sample, index=False)

    df = pd.read_csv(sample)

    # 1) Health check
    analysis = step("analyze (health check)", lambda: dataset_analyzer.analyze_dataframe(df))
    print(f"  rows={analysis['rows']} cols={analysis['columns']} warnings={len(analysis['warnings'])}")

    # 2) Target detection
    detection = step("detect target & task", lambda: __import__('app.services.target_detector', fromlist=['x']).detect_target_and_task(df))
    print(f"  target={detection['suggested_target']} task={detection['task']} conf={detection['confidence']}")
    assert detection["suggested_target"] == "Churn" or detection["suggested_target"] == "churn", "should find churn target"
    # ID column should NOT be suggested
    assert detection["suggested_target"] != "customerID"

    # 3) SDV generation (small, GaussianCopula for speed)
    gen = step("SDV generate (GaussianCopula, 300 rows)", lambda: sdv_generator.generate_full(
        df.drop(columns=["customerID"]), n_rows=300, synthesizer_override="GaussianCopula"))
    print(f"  approach={gen['approach']} fit={gen['fit_seconds']}s sample={gen['sample_seconds']}s warnings={gen['warnings']}")
    assert len(gen["dataframe"]) == 300

    # 4) Evaluation
    evalr = step("evaluate real vs synthetic", lambda: evaluation_engine.evaluate(df, gen["dataframe"], target="Churn"))
    score = evalr["data_nova_score"]["value"]
    print(f"  score={score} corr_mad={evalr['correlation_comparison'].get('mean_abs_difference')}")
    assert score is not None and 0 <= score <= 1

    # 5) Model recommendation
    rec = step("recommend model", lambda: model_recommender.recommend_model(df, "classification", "Churn"))
    print(f"  model={rec['model_key']}")

    # 6) ML utility
    mlr = step("ML utility experiments", lambda: ml_utility_engine.run_ml_utility(
        df, gen["dataframe"], target="Churn", task="classification", model_key=rec["model_key"]))
    print(f"  A.acc={mlr['experiment_a']['metrics']['accuracy']} B.acc={mlr['experiment_b']['metrics']['accuracy']}")

    # 7) Explainability
    expl = step("explain results", lambda: {
        "evaluation": explainability_engine.explain_evaluation(evalr),
        "ml_utility": explainability_engine.explain_ml_utility(mlr),
        "generation": explainability_engine.explain_generation(gen["approach"], gen["reason"], gen["warnings"]),
    })
    print(f"  findings: eval={len(expl['evaluation'])} ml={len(expl['ml_utility'])}")

    # 8) Requirement parsing (Mode B)
    spec = step("parse NL requirements", lambda: requirement_parser.parse_requirements(
        "I need a classification dataset with 10000 records for predicting customer churn. "
        "Include age, income, tenure, monthly charges and contract type. "
        "The target should be churn with approximately 20% positive cases."))
    sd = requirement_parser.spec_to_dict(spec)
    n_cols = len(sd.get("columns") or sd.get("features") or [])
    print(f"  rows={sd['rows']} task={sd['task']} target={sd['target']} columns={n_cols} clar={len(sd['clarifications'])}")
    assert sd["rows"] == 10000 and sd["task"] == "classification" and sd["target"] == "churn"
    assert sd["class_distribution"] and sd["class_distribution"].get("positive_pct") == 20

    # 9) Spec-based generation (Mode B) — small for speed; use v2 semantic columns
    spec.rows = 500
    spec.columns = (spec.columns or spec.features or [])[:3]
    spec.features = spec.columns
    if spec.columns and "columns" in sd:
        for c in spec.columns:
            c.setdefault("technical_type", "float")
            c.setdefault("semantic_type", "generic_text")
            c.setdefault("min", None); c.setdefault("max", None); c.setdefault("categories", None)
            c.setdefault("distribution", None); c.setdefault("unique", False); c.setdefault("detected", False)
    genb = step("generate from spec", lambda: spec_synthesizer.generate_from_spec(spec))
    bdf = genb["dataframe"]
    print(f"  rows={len(bdf)} target_share={bdf['churn'].mean():.2f}")
    assert abs(bdf["churn"].mean() - 0.20) < 0.02, "class distribution should honor 20%"

    # 10) Persistence
    step("save project to SQLite", lambda: database.save_project({
        "name": "Smoke test project", "mode": "upload", "rows_real": len(df), "columns": df.shape[1],
        "target": "Churn", "task": "classification", "synthesizer": gen["approach"],
        "rows_synthetic": len(gen["dataframe"]), "fit_seconds": gen["fit_seconds"],
        "evaluation": __import__('json').dumps(evalr), "ml_utility": __import__('json').dumps(mlr),
    }))
    projects = database.list_projects()
    print(f"  saved projects: {len(projects)}")
    assert len(projects) >= 1

    # 11) PDF report
    proj = database.get_project(projects[0]["id"])
    out = Path(__file__).resolve().parent.parent / "data" / "reports" / "smoke_test_report.pdf"
    step("generate PDF report", lambda: report_generator.generate_report(proj, out))
    print(f"  report size: {out.stat().st_size} bytes")

    print()
    for mark, name in results:
        print(f"{mark} {name}")
    print("\nALL SMOKE TESTS PASSED" if all(m == PASS for m, _ in results) else "\nSOME TESTS FAILED")


if __name__ == "__main__":
    main()

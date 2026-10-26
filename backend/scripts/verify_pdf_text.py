"""
Verify that a DataNova PDF report contains the expected values from the
stored project payloads. Usage:

    python -X utf8 scripts/verify_pdf_text.py <project_id>

Decodes ReportLab streams (zlib or ASCII85+Flate), extracts every text-show
operator, and checks the report against the canonical SQLite project record.
This is the source-of-truth check: UI and PDF must agree.
"""
from __future__ import annotations

import base64
import html
import re
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import database  # noqa: E402

TJ = re.compile(rb"\(((?:[^()\\]|\\.)*)\)\s*Tj")
TJ_ARRAY = re.compile(rb"\[(.*?)\]\s*TJ", re.S)
LIT = re.compile(rb"\(((?:[^()\\]|\\.)*)\)")


OCTAL = re.compile(rb"\\([0-7]{1,3})")
ESCAPES = {b"n": b"\n", b"r": b"\r", b"t": b"\t", b"(": b"(", b")": b")", b"\\": b"\\"}


def _unescape_pdf_string(s: bytes) -> bytes:
    s = OCTAL.sub(lambda m: bytes([int(m.group(1), 8)]), s)
    for k, v in ESCAPES.items():
        s = s.replace(b"\\" + k, v)
    return s


def extract_pdf_text(pdf_path: Path) -> str:
    raw = pdf_path.read_bytes()
    out: list[bytes] = []
    for m in re.finditer(rb">>\s*stream\r?\n(.*?)endstream", raw, re.S):
        body = m.group(1)
        data = None
        try:
            data = zlib.decompress(body)
        except Exception:
            try:
                data = zlib.decompress(base64.a85decode(body, adobe=True))
            except Exception:
                continue
        out.extend(_unescape_pdf_string(t) for t in TJ.findall(data))
        for arr in TJ_ARRAY.findall(data):
            out.extend(_unescape_pdf_string(t) for t in LIT.findall(arr))
    return html.unescape(b" ".join(out).decode("latin-1", "ignore"))


def main() -> None:
    project_id = int(sys.argv[1]) if len(sys.argv) > 1 else 17
    proj = database.get_project(project_id)
    report_path = Path(proj["report_path"] or "")
    if not report_path.exists():
        # Reports are named after the project (e.g. my-project-20260927-report.pdf).
        reports_dir = Path(__file__).resolve().parent.parent / "data" / "reports"
        candidates = sorted(reports_dir.glob("*-report.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            print("FAIL no report file found for project", project_id)
            sys.exit(1)
        report_path = candidates[0]

    text = extract_pdf_text(report_path)
    print(f"report: {report_path.name} | extracted chars: {len(text)}")

    checks: dict[str, bool] = {}

    def fmt_like_report(x) -> str:
        """Mirror report_generator._fmt so numbers compare exactly."""
        try:
            v = float(x)
        except (TypeError, ValueError):
            return str(x)
        s = f"{v:.4f}".rstrip("0").rstrip(".")
        return s if s not in ("", "-0") else "0"

    def add(label: str, needle, require=True):
        if needle is None:
            return
        if isinstance(needle, float):
            needle = fmt_like_report(needle)
        # The extractor sees WinAnsi bytes (em-dash -> 0x97); compare ASCII-only
        # and collapse whitespace (text chunks are joined with spaces).
        def norm(s: str) -> str:
            return re.sub(r"\s+", " ", s.encode("ascii", "ignore").decode())
        norm_text = norm(text)
        norm_needle = norm(str(needle))
        if not norm_needle:
            return
        checks[label] = (norm_needle in norm_text) if require else (norm_needle not in norm_text)

    # 1. project information
    add("project name", proj["name"])
    if proj.get("dataset_filename"):
        add("dataset filename", proj["dataset_filename"])
    if proj.get("rows_real"):
        add("original rows", proj["rows_real"])
    if proj.get("columns"):
        add("original columns", proj["columns"])

    # 2. dataset health
    analysis = proj.get("analysis") or {}
    if analysis:
        add("missing cells", analysis.get("total_missing_cells"))
        add("duplicate rows", analysis.get("duplicate_rows"))

    # 3. generation configuration
    add("synthesizer name", proj.get("synthesizer"))
    if proj.get("rows_synthetic"):
        add("synthetic rows", proj["rows_synthetic"])
    req_spec = proj.get("requirement_spec") or {}
    if req_spec.get("raw_text"):
        add("user requirement quote", req_spec["raw_text"][:25])

    # 5-8. evaluation values
    ev = proj.get("evaluation") or {}
    if ev:
        score = (ev.get("data_nova_score") or {}).get("value")
        add("DataNova Score value", score)
        add("score NOT-standard disclaimer", "NOT an industry-standard", require=True)
        for c in (ev.get("column_results") or [])[:6]:
            add(f"column {c.get('column')}", c.get("column"))
            st = c.get("stats")
            if st and st.get("mean", {}).get("real") is not None:
                add(f"mean of {c.get('column')}", float(st["mean"]["real"]))
        tgt = ev.get("target_comparison") or {}
        if tgt.get("target"):
            add("target column", tgt.get("target"))
        for lbl, comp in (("statistical component", (ev.get("data_nova_score") or {}).get("components", {}).get("statistical")),
                          ("correlation component", (ev.get("data_nova_score") or {}).get("components", {}).get("correlation")),
                          ("target component", (ev.get("data_nova_score") or {}).get("components", {}).get("target"))):
            add(lbl, comp)

    # 9-10. ML utility values
    ml = proj.get("ml_utility") or {}
    if ml:
        add("model label", (ml.get("model") or {}).get("label"))
        a = (ml.get("experiment_a") or {}).get("metrics") or {}
        b = (ml.get("experiment_b") or {}).get("metrics") or {}
        keys = ["accuracy", "precision", "recall", "f1"] if ml.get("task") == "classification" else ["mae", "rmse", "r2"]
        for k in keys:
            if a.get(k) is not None:
                add(f"experiment A {k}", float(a[k]))
            if b.get(k) is not None:
                add(f"experiment B {k}", float(b[k]))
        add("experiment labels", "real-trained")
        add("experiment labels 2", "synthetic-trained")

    # branding, mode label, section structure
    add("brand line", "Synthetic Data Studio")
    add("project report cover", "PROJECT REPORT")
    mode = proj.get("mode")
    if mode == "upload":
        add("mode label", "Expanded Dataset (Mode A)")
    elif mode == "requirements":
        add("mode label", "From Requirements (Mode B)")
    if proj.get("messy_file"):
        add("messy section", "10. Messy data configuration")
        add("messy file named", str(proj.get("messy_file")))
    add("overview section", "1. Overview")
    add("no academic wording", "academic", require=False)
    add("no TY BSc wording", "TY BSc", require=False)

    # score, interpretation, limitations
    add("DataNova Score section", "DataNova Score")
    add("limitations section", "does not automatically guarantee privacy")
    add("version metadata", "1.0.0")
    add("DATANOVA header band", "DATANOVA")
    add("report title", "Synthetic Data Evaluation Report")

    failed = [k for k, v in checks.items() if not v]
    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k)
    total = len(checks)
    print(f"{total - len(failed)}/{total} PDF content checks passed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

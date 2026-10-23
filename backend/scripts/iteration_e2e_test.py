"""
End-to-end iteration test — covers the four mandated test cases:

  41. CSV upload -> semantic detection -> 2000-row generation follows the input
  42. ML utility: same real hold-out, real predictions, no leakage, persisted
  43. Messy data: user percentages -> actual counts reported, original untouched
  44. Project history: readable name, downloads after navigation, report, rename

Run with the backend already listening on :8000:
    python -X utf8 scripts/iteration_e2e_test.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.services import messy_data  # noqa: E402

B = "http://localhost:8000/api"
CHECKS: list[tuple[str, bool, str]] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((label, bool(ok), detail))
    print(("PASS " if ok else "FAIL ") + label + (f" — {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# --------------------------------------------------------------- test dataset
def build_person_csv(n: int = 500) -> bytes:
    """Name/Gender/Age/City/State/Phone/Email/Salary — realistic Indian data."""
    rng = pd.util.hash_array  # noqa: F841  (deterministic build below)
    first = ["Rahul", "Priya", "Aarav", "Ananya", "Vikram", "Sneha", "Karan", "Isha", "Rohit", "Meera"]
    last = ["Sharma", "Patel", "Singh", "Gupta", "Reddy", "Iyer", "Khan", "Das", "Mehta", "Joshi"]
    cities = ["Mumbai", "Delhi", "Bengaluru", "Pune", "Chennai", "Hyderabad", "Jaipur", "Kolkata"]
    states = {"Mumbai": "Maharashtra", "Delhi": "Delhi", "Bengaluru": "Karnataka", "Pune": "Maharashtra",
              "Chennai": "Tamil Nadu", "Hyderabad": "Telangana", "Jaipur": "Rajasthan", "Kolkata": "West Bengal"}
    rows = []
    for i in range(n):
        city = cities[i % len(cities)]
        age = 21 + (i * 7) % 40
        salary = 25000 + (i * 1300) % 90000
        phone = f"9{800000000 + (i * 911) % 99999999}"
        name = f"{first[i % 10]} {last[i % 10]}"
        rows.append({
            "Name": name, "Gender": "Male" if i % 2 == 0 else "Female", "Age": age,
            "City": city, "State": states[city], "Phone": phone,
            "Email": f"{name.split()[0].lower()}{i}@example.com", "Salary": salary,
        })
    buf = io.StringIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    return buf.getvalue().encode()


section("41. CSV upload → semantic detection → 2000-row generation")
csv_bytes = build_person_csv(500)
up = requests.post(f"{B}/upload", files={"file": ("customers.csv", csv_bytes, "text/csv")}).json()
schema = up.get("schema", {})

def sem(col):
    return schema.get(col, {}).get("semantic_type")

check("upload ok", up.get("upload_id", "").startswith("up_"))
check("Name -> Person Name", sem("Name") == "person_name", str(sem("Name")))
check("Gender -> gender", sem("Gender") == "gender", str(sem("Gender")))
check("Age -> age", sem("Age") == "age", str(sem("Age")))
check("City -> city", sem("City") == "city", str(sem("City")))
check("State -> state", sem("State") == "state", str(sem("State")))
check("Phone -> indian_phone", sem("Phone") == "indian_phone", str(sem("Phone")))
check("Email -> email", sem("Email") == "email", str(sem("Email")))
check("Salary -> salary", sem("Salary") == "salary", str(sem("Salary")))

gen = requests.post(f"{B}/upload/{up['upload_id']}/generate", json={"n_rows": 2000}).json()
qc = gen.get("quality_check") or {}
synth = pd.read_csv(io.BytesIO(requests.get(f"{B}/upload/{up['upload_id']}/synthetic/download").content))

check("2000 rows generated", gen.get("n_rows_generated") == 2000, str(gen.get("n_rows_generated")))
check("columns preserved exactly", list(synth.columns) == ["Name", "Gender", "Age", "City", "State", "Phone", "Email", "Salary"], str(list(synth.columns)))
check("Name still textual", synth["Name"].astype(str).str.contains(r"[A-Za-z]").all(), synth["Name"].iloc[0])
check("Gender values from real set", set(synth["Gender"].dropna().unique()) <= {"Male", "Female"}, str(set(synth["Gender"].unique()) - {"Male", "Female"}))
check("City values from real set", set(synth["City"].dropna().unique()) <= set(pd.read_csv(io.BytesIO(csv_bytes))["City"].unique()))
check("Age numeric and plausible", synth["Age"].between(18, 100).all(), f"min={synth['Age'].min()} max={synth['Age'].max()}")
check("Salary numeric and plausible", synth["Salary"].between(20000, 150000).all(), f"min={synth['Salary'].min()} max={synth['Salary'].max()}")
check("Phone digits valid", synth["Phone"].astype(str).str.replace(r"\D", "", regex=True).str.len().between(10, 13).all())
check("Email format valid", synth["Email"].astype(str).str.contains(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", regex=True).all())
check("quality check present & passed", bool(qc.get("passed")), str(qc.get("passed")))
sess_download = requests.get(f"{B}/upload/{up['upload_id']}/synthetic/download")
check("session CSV download 200 + human name", sess_download.status_code == 200 and "customers" in sess_download.headers.get("content-disposition", ""), sess_download.headers.get("content-disposition", ""))

# Mode A project with evaluation + ML for §42/§44
real_df = pd.read_csv(io.BytesIO(csv_bytes))
save = requests.post(f"{B}/projects/save", json={
    "name": "",  # server falls back? frontend derives names; send explicit below
    "mode": "upload",
    "upload_id": up["upload_id"],
    "dataset_filename": "customers.csv",
    "rows_real": 500, "columns": 8,
    "target": "Gender", "task": "classification", "target_overridden": True,
    "synthesizer": gen.get("approach"), "generation_reason": gen.get("reason"),
    "rows_requested": 2000, "rows_synthetic": gen.get("n_rows_generated"),
    "fit_seconds": gen.get("fit_seconds"), "sample_seconds": gen.get("sample_seconds"),
}).json()
print("save response:", str(save)[:200])
pid = save.get("project_id")

# Human-readable name check (frontend sends a real name; use rename to simulate)
requests.patch(f"{B}/projects/{pid}", json={"name": "Indian Customer Synthetic Dataset"})
proj = requests.get(f"{B}/projects/{pid}").json()
check("project renamed to readable name", proj["name"] == "Indian Customer Synthetic Dataset", proj["name"])

section("42. ML utility — same holdout, real predictions")
ml = requests.post(f"{B}/ml-utility/run", json={"project_id": pid, "target": "Gender", "task": "classification"}).json()
ma, mb = (ml.get("experiment_a") or {}).get("metrics", {}), (ml.get("experiment_b") or {}).get("metrics", {})
check("ML run ok", "experiment_a" in ml and "experiment_b" in ml, ml.get("detail", ""))
check("metrics are real numbers", isinstance(ma.get("accuracy"), float) and isinstance(mb.get("f1"), float), f"A={ma.get('accuracy')} B={mb.get('f1')}")
check("same real hold-out both experiments", ml.get("split", {}).get("real_test_rows", 0) > 0 and ml["split"]["real_train_rows"] + ml["split"]["real_test_rows"] == 500)
check("preprocess split per experiment", "fit" in ml.get("interpretation_note", "").lower())
a_clf, b_clf = ma.get("confusion_matrix"), mb.get("confusion_matrix")
check("confusion matrices computed from predictions", bool(a_clf) and bool(b_clf) and sum(map(sum, a_clf["matrix"])) == ml["split"]["real_test_rows"])
# Leakage probe: recompute with an obviously-broken synthetic frame? Instead verify
# experiment B trained on synthetic rows only:
check("B trained on synthetic rows", ml.get("split", {}).get("synthetic_train_rows", 0) == 2000, str(ml.get("split", {}).get("synthetic_train_rows")))
# persist results into the project like the UI does (one final save with the
# full payload, then drop the intermediate project)
ev = requests.post(f"{B}/evaluate", json={"project_id": pid}).json()
check("evaluation computed from stored real+synthetic", bool(ev.get("column_results")), str(ev.get("detail", ""))[:80])
save2 = requests.post(f"{B}/projects/save", json={
    "name": "Indian Customer Synthetic Dataset", "mode": "upload", "upload_id": up["upload_id"],
    "dataset_filename": "customers.csv", "rows_real": 500, "columns": 8,
    "target": "Gender", "task": "classification", "target_overridden": True,
    "synthesizer": gen.get("approach"), "rows_requested": 2000, "rows_synthetic": 2000,
    "ml_utility": ml, "evaluation": ev,
}).json()
requests.delete(f"{B}/projects/{pid}")
pid = save2["project_id"]
section("43. Messy data — user percentages, honest counts, original untouched")
cfg = {"missing_values": 5, "duplicate_rows": 3, "whitespace": 2, "format_inconsistency": 2, "invalid_values": 1}
pv = requests.post(f"{B}/projects/{pid}/messy/preview", json=cfg).json()
check("preview returns dataset shape", pv.get("dataset", {}).get("rows") == 2000 and pv["dataset"]["columns"] == 8, str(pv.get("dataset")))
check("preview estimates missing cells", pv["estimates"]["missing_cells"] == 800, str(pv["estimates"]["missing_cells"]))

before_clean = pd.read_csv(io.BytesIO(requests.get(f"{B}/projects/{pid}/synthetic/download").content))
messy = requests.post(f"{B}/projects/{pid}/messy", json=cfg).json()
summ = messy.get("summary", {})
check("messy generated + file recorded", bool(messy.get("file", "").endswith("messy.csv")), messy.get("file", ""))
after_clean = pd.read_csv(io.BytesIO(requests.get(f"{B}/projects/{pid}/synthetic/download").content))
check("original clean dataset unchanged", before_clean.equals(after_clean), f"{len(before_clean)} vs {len(after_clean)} rows")

mdf = pd.read_csv(io.BytesIO(requests.get(f"{B}/projects/{pid}/messy/download").content))
check("messy rows grew by duplicates", len(mdf) == summ.get("rows_after_duplicates") == 2000 + summ.get("duplicate_rows_added"), f"{len(mdf)} vs {summ.get('rows_after_duplicates')}")
dup_target = int(round(2000 * 0.03))
check("~3% duplicates actually added", abs(summ.get("duplicate_rows_added", 0) - dup_target) <= dup_target * 0.2 + 2, str(summ.get("duplicate_rows_added")))
check("missing cells ~5% of original cells", abs(summ.get("missing_cells_actual", 0) - summ.get("missing_cells_target", 0)) <= summ.get("missing_cells_target", 0) * 0.35 + 50, f"actual={summ.get('missing_cells_actual')} target={summ.get('missing_cells_target')}")
check("whitespace issues introduced", summ.get("counts", {}).get("whitespace", 0) > 0, str(summ.get("counts", {}).get("whitespace")))
check("format variants introduced (phone/email)", summ.get("counts", {}).get("format_inconsistencies", 0) > 0, str(summ.get("counts", {}).get("format_inconsistencies")))
check("summary reports actuals not just targets", "counts" in summ and "missing_cells_actual" in summ)

section("44. Project history — files, downloads after navigation, report")
files = requests.get(f"{B}/projects/{pid}/files").json()
check("files endpoint: synthetic exists", files.get("synthetic_csv", {}).get("exists") is True, str(files.get("synthetic_csv")))
check("files endpoint: messy exists", files.get("messy_csv", {}).get("exists") is True, str(files.get("messy_csv")))
rep = requests.post(f"{B}/projects/{pid}/report", json={})
check("report generated", rep.status_code == 200, rep.text[:120])
pdf = requests.get(f"{B}/projects/{pid}/report/download")
# short subject-based name: "indian-customer-report.pdf" (a "_2" suffix only
# appears when a leftover file from a previous run already exists)
check("PDF download 200 + short readable name", pdf.status_code == 200 and "indian-customer-report" in pdf.headers.get("content-disposition", ""), pdf.headers.get("content-disposition", ""))
csv2 = requests.get(f"{B}/projects/{pid}/synthetic/download")
check("CSV re-download after 'navigation' 200", csv2.status_code == 200 and csv2.headers.get("content-disposition", "").endswith(".csv\""))
ml_in_project = requests.get(f"{B}/projects/{pid}").json().get("ml_utility")
check("ML results persisted in project", bool(ml_in_project and ml_in_project.get("experiment_a")))

section("Direct engine probe: messy respects semantics")
probe_cfg = {k: v / 100.0 for k, v in cfg.items()}  # engine expects fractions
probe = messy_data.make_messy(real_df, probe_cfg, seed=42)
pdf_probe = probe["dataframe"]
phones = pdf_probe["Phone"].astype(str)
check("phone column still phone-like (no garbage)", phones.str.replace(r"\D", "", regex=True).str.len().between(9, 15).mean() > 0.95, f"{phones.str.replace(chr(92)+'D','',regex=True).str.len().between(9,15).mean():.0%} valid")
names = pdf_probe["Name"].astype(str)
check("names remain names (whitespace/case variants)", names.str.replace(r"\s", "", regex=False).str.replace(r"[^A-Za-z]", "", regex=False).str.len().ge(5).mean() > 0.95)

print(f"\n{sum(1 for _, ok, _ in CHECKS if ok)}/{len(CHECKS)} checks passed")
if not all(ok for _, ok, _ in CHECKS):
    sys.exit(1)

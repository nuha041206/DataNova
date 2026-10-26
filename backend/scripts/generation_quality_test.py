"""
Generation-accuracy test battery — tests 1-7 from the quality iteration.

Verifies ACTUAL downloaded CSV output for every scenario, not UI state.
Run with the backend listening on :8000:
    python -X utf8 scripts/generation_quality_test.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

B = "http://localhost:8000/api"
CHECKS: list[tuple[str, bool, str]] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((label, bool(ok), detail))
    print(("PASS " if ok else "FAIL ") + label + (f" — {detail}" if detail and not ok else ""))


def section(t: str) -> None:
    print(f"\n=== {t} ===")


def upload(df: pd.DataFrame, name: str) -> str:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return requests.post(f"{B}/upload", files={"file": (name, buf.getvalue().encode(), "text/csv")}).json()["upload_id"]


def generate(uid: str, n: int) -> dict:
    return requests.post(f"{B}/upload/{uid}/generate", json={"n_rows": n}).json()


def synth_df(uid: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(requests.get(f"{B}/upload/{uid}/synthetic/download").content))


# ---------------------------------------------------------------- TEST 1
section("TEST 1 — mixed semantics (Name/Gender/Age/Address/Phone/Salary)")
df1 = pd.DataFrame({
    "Name": [f"Person {i}" for i in range(200)],
    "Gender": ["Male" if i % 2 else "Female" for i in range(200)],
    "Age": [25 + (i % 30) for i in range(200)],
    "Address": [f"{12 + i} MG Road, Bengaluru" for i in range(200)],
    "Phone Number": [f"98{10000000 + i}" for i in range(200)],
    "Salary": [30000 + i * 500 for i in range(200)],
})
uid = upload(df1, "people.csv")
g = generate(uid, 2000)
s = synth_df(uid)
check("T1 rows", len(s) == 2000, str(len(s)))
check("T1 columns preserved", list(s.columns) == list(df1.columns))
check("T1 names textual", s["Name"].astype(str).str.contains("[A-Za-z]").all())
check("T1 gender from real pool", set(s["Gender"].dropna().unique()) <= {"Male", "Female"})
check("T1 age in plausible range", s["Age"].between(20, 60).all(), f"{s['Age'].min()}-{s['Age'].max()}")
check("T1 phones valid", s["Phone Number"].astype(str).str.replace(r"\D", "", regex=True).str.len().between(10, 13).all())
check("T1 salary within 10% of source range", s["Salary"].between(df1.Salary.min() * 0.9, df1.Salary.max() * 1.1).all(),
      f"{s['Salary'].min()}-{s['Salary'].max()} vs {df1.Salary.min()}-{df1.Salary.max()}")
check("T1 quality check passed", (g.get("quality_check") or {}).get("passed") is True)
check("T1 new records (not copies)",
      s.astype(str).agg(" | ".join, axis=1).nunique() > 1900
      and not set(map(tuple, s.astype(str).values)) & set(map(tuple, df1.astype(str).values)))

# ---------------------------------------------------------------- TEST 2
section("TEST 2 — classification dataset (target + ML utility)")
df2 = pd.DataFrame({
    "Age": [22 + (i * 5) % 40 for i in range(600)],
    "Tenure": [1 + (i * 3) % 24 for i in range(600)],
    "MonthlyCharges": [20 + (i * 7) % 100 for i in range(600)],
    "Contract": ["Month-to-month" if i % 3 else "Two year" for i in range(600)],
    "Churn": ["Yes" if (i % 3 == 0 and i % 2) else "No" for i in range(600)],
})
uid2 = upload(df2, "churn.csv")
g2 = generate(uid2, 2000)
s2 = synth_df(uid2)
check("T2 rows", len(s2) == 2000)
check("T2 churn values preserved", set(s2["Churn"].dropna().unique()) <= {"Yes", "No"})
check("T2 quality passed", (g2.get("quality_check") or {}).get("passed") is True)
ml = requests.post(f"{B}/ml-utility/run", json={"upload_id": uid2, "target": "Churn", "task": "classification"}).json()
ma = (ml.get("experiment_a") or {}).get("metrics") or {}
mb = (ml.get("experiment_b") or {}).get("metrics") or {}
check("T2 ML runs with real metrics", isinstance(ma.get("accuracy"), float) and isinstance(mb.get("accuracy"), float),
      f"A={ma.get('accuracy')} B={mb.get('accuracy')}")
check("T2 same holdout for A and B", ml.get("split", {}).get("real_test_rows", 0) == 120)
check("T2 imbalance info present", "imbalance" in ml)

# ---------------------------------------------------------------- TEST 3
section("TEST 3 — numerical + categorical mix")
df3 = pd.DataFrame({
    "Department": ["HR", "Finance", "IT", "Sales"] * 125,
    "Experience": [2 + (i * 2) % 20 for i in range(500)],
    "Score": [50 + (i * 13) % 50 for i in range(500)],
    "Level": ["Junior", "Senior"] * 250,
})
uid3 = upload(df3, "employees.csv")
g3 = generate(uid3, 1500)
s3 = synth_df(uid3)
check("T3 rows", len(s3) == 1500)
check("T3 categorical pools preserved", set(s3["Department"]) <= {"HR", "Finance", "IT", "Sales"} and set(s3["Level"]) <= {"Junior", "Senior"})
check("T3 numeric ranges respected", s3["Experience"].between(0, 25).all() and s3["Score"].between(45, 105).all(),
      f"exp {s3['Experience'].min()}-{s3['Experience'].max()}, score {s3['Score'].min()}-{s3['Score'].max()}")
check("T3 quality passed", (g3.get("quality_check") or {}).get("passed") is True)

# ---------------------------------------------------------------- TEST 4
section("TEST 4 — dataset with missing values")
df4 = pd.DataFrame({
    "City": ["Mumbai", "Delhi", None, "Pune"] * 125,
    "Age": [30 + (i % 20) if i % 7 else None for i in range(500)],
    "Income": [20000 + i * 100 if i % 5 else None for i in range(500)],
})
uid4 = upload(df4, "gaps.csv")
g4 = generate(uid4, 1000)
s4 = synth_df(uid4)
check("T4 rows", len(s4) == 1000)
check("T4 generation handles missingness", (g4.get("quality_check") or {}).get("passed") is True,
      str([c for c in (g4.get("quality_check") or {}).get("checks", []) if not c["passed"]])[:120])
check("T4 missing-rate roughly preserved (within 0-15%)",
      0.0 <= s4.isna().mean().mean() <= 0.15, f"{s4.isna().mean().mean():.1%}")

# ---------------------------------------------------------------- TEST 5
section("TEST 5 — dataset with duplicate records")
df5 = pd.concat([df1.head(100), df1.head(100)], ignore_index=True)
uid5 = upload(df5, "dupes.csv")
g5 = generate(uid5, 800)
s5 = synth_df(uid5)
check("T5 rows", len(s5) == 800)
check("T5 synthetic rows not locked to few combos", s5.astype(str).drop_duplicates().shape[0] > 700,
      f"{s5.astype(str).drop_duplicates().shape[0]} unique of 800")
check("T5 quality passed", (g5.get("quality_check") or {}).get("passed") is True)

# ---------------------------------------------------------------- TEST 6
section("TEST 6 — Mode 2 natural language → CSV satisfies request")
spec = requests.post(f"{B}/requirements/parse", json={"text":
    "Create 2000 unique customer records with name, gender, age from 18 to 65, Indian phone number, address and salary."
}).json()["spec"]
gen6 = requests.post(f"{B}/requirements/generate", json={"spec": spec, "seed": 42}).json()
s6 = pd.read_csv(io.BytesIO(requests.get(f"{B}/upload/{gen6['session_id']}/synthetic/download").content))
check("T6 parsed rows", spec["rows"] == 2000 and spec["unique"] is True)
check("T6 parsed columns + age range",
      [c["name"] for c in spec["columns"]] == ["Name", "Gender", "Age", "Phone", "Address", "Salary"]
      and next(c for c in spec["columns"] if c["name"] == "Age")["min"] == 18,
      str([c["name"] for c in spec["columns"]]))
check("T6 CSV rows = 2000", len(s6) == 2000, str(len(s6)))
check("T6 CSV columns match spec order", list(s6.columns) == [c["name"] for c in spec["columns"]])
check("T6 all rows unique", int(s6.duplicated().sum()) == 0, f"{int(s6.duplicated().sum())} duplicates")
check("T6 age within 18-65", s6["Age"].between(18, 65).all(), f"{s6['Age'].min()}-{s6['Age'].max()}")
check("T6 Indian phone format", s6["Phone"].astype(str).str.replace(r"\D", "", regex=True).str.len().eq(10).all()
      and s6["Phone"].astype(str).str.replace(r"\D", "", regex=True).str.startswith(("6", "7", "8", "9")).mean() > 0.9)
check("T6 salary numeric", pd.api.types.is_numeric_dtype(s6["Salary"]))

# ---------------------------------------------------------------- TEST 7
section("TEST 7 — Mode 1 + messy-data settings (actual % respected)")
m = requests.post(f"{B}/upload/{uid}/messy", json={
    "missing_values": 5, "duplicate_rows": 3, "whitespace": 2,
    "format_inconsistency": 2, "invalid_values": 1,
}).json()
summ = m["summary"]
clean = synth_df(uid)
messy = pd.read_csv(io.BytesIO(requests.get(f"{B}/upload/{uid}/messy/download").content))
check("T7 clean dataset untouched", len(clean) == 2000 and clean.isna().sum().sum() == 0)
check("T7 messy rows grew by 3%", len(messy) == 2060, str(len(messy)))
check("T7 duplicates ≈ 3%", abs(summ["duplicate_rows_added"] - 60) <= 3, str(summ["duplicate_rows_added"]))
check("T7 actual missing % reported and ≈ 5% of cells",
      0 < m["actual_percentages"]["missing_values"] <= 6, str(m["actual_percentages"]))
check("T7 messy file has missing values", messy.isna().sum().sum() > 0, str(int(messy.isna().sum().sum())))
check("T7 whitespace + format variants applied",
      summ["counts"]["whitespace"] > 0 and summ["counts"]["format_inconsistencies"] > 0)

# preview/download consistency spot-check (§11) — compare numerically so
# JSON '3' vs CSV-parsed 3.0 counts as the same value.
prev = requests.get(f"{B}/upload/{uid}/synthetic-preview?n=5").json()["rows"]

def _same(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-9
    except (TypeError, ValueError):
        return str(a) == str(b)

head = s.head(5)
check("§11 preview matches downloaded CSV", len(prev) == 5 and
      all(_same(prev[i][c], head.iloc[i][c]) or (pd.isna(prev[i][c]) and pd.isna(head.iloc[i][c]))
          for i in range(5) for c in list(s.columns)[:3]))

print(f"\n{sum(1 for _, ok, _ in CHECKS if ok)}/{len(CHECKS)} checks passed")
sys.exit(0 if all(ok for _, ok, _ in CHECKS) else 1)

"""
DataNova accuracy & input-handling test (no server required).

Covers, with measurable checks:
  A. CSV upload validation (blank, header-only, blank cells, duplicate headers,
     inconsistent columns, empty headers, binary, single column, wrong type)
  B. Natural-language requirement parsing (valid prompts, spelling mistakes,
     gibberish, empty input, unfamiliar column names, explicit row counts)
  C. Generation accuracy (requested schema, formats, ranges, row counts,
     cross-column consistency, no missing/duplicate rows)
  D. Regression of the documented example prompts
  E. Mode A data quality via the real endpoints (ID-like Age/Address stays
     human-readable, name diversity, DOB~Age / hire>=DOB / city~state rules)
  F. PDF report generation (None-quantile crash shape, real project report)
  G. Download file names (short, meaningful, collision-safe, legacy files)

Run:  python -X utf8 scripts/accuracy_input_test.py
Every check prints PASS/FAIL with its measured value; a summary follows.
"""
from __future__ import annotations

import io
import json
import random
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from app import config, database
from app.services import (
    dataset_analyzer,
    generation_validator,
    requirement_parser,
    semantic_generators,
    spec_synthesizer,
)
from app.services.dataset_analyzer import DatasetError
from app.services.report_generator import generate_report as build_report

# PDF text extraction is shared with the source-of-truth script
# scripts/verify_pdf_text.py (same implementation, no new dependency).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_pdf_text import extract_pdf_text  # noqa: E402

PASS, FAIL = "PASS", "FAIL"
results: list = []


def check(name: str, fn):
    """Run one check; never aborts the run."""
    try:
        detail = fn()
        results.append((PASS, name))
        print(f"  PASS  {name}" + (f"  [{detail}]" if detail else ""))
        return True
    except Exception as exc:
        results.append((FAIL, f"{name}"))
        print(f"  FAIL  {name}  -> {type(exc).__name__}: {exc}")
        return False


def expect_dataset_error(name: str, raw: bytes, filename: str, contains: str = "") -> bool:
    """The upload must be REJECTED with a clear DatasetError message."""
    def _run():
        try:
            dataset_analyzer.load_csv_from_bytes(raw, filename)
        except DatasetError as exc:
            msg = str(exc)
            assert msg.strip(), "error message must not be empty"
            if contains:
                assert contains.lower() in msg.lower(), f"message should mention '{contains}', got: {msg}"
            return f"rejected: {msg[:80]}"
        raise AssertionError("file was accepted but should have been rejected")
    return check(name, _run)


# ---------------------------------------------------------------------------
# A. CSV upload validation
# ---------------------------------------------------------------------------
def test_csv_uploads():
    print("\n== A. CSV upload validation ==")

    # A1 valid CSV with normal headers
    def a1():
        df = dataset_analyzer.load_csv_from_bytes(
            b"name,age,city\nAarav,25,Mumbai\nDiya,31,Pune\n", "ok.csv")
        assert list(df.columns) == ["name", "age", "city"], list(df.columns)
        assert df.shape == (2, 3)
        return f"shape={df.shape} columns={list(df.columns)}"
    check("A1 valid CSV with normal headers is accepted", a1)

    # A2 completely blank file
    expect_dataset_error("A2 completely empty file is rejected", b"", "empty.csv", contains="empty")

    # A2b file with only blank lines / whitespace
    expect_dataset_error("A2b blank-lines-only file is rejected",
                         b"\n\n   \n\n\t\n", "blank.csv")

    # A3 header-only CSV
    expect_dataset_error("A3 header-only CSV is rejected",
                         b"name,age,city\n", "header_only.csv", contains="no data rows")

    # A4 CSV with blank cells (must still load; health check must see them)
    def a4():
        df = dataset_analyzer.load_csv_from_bytes(
            b"name,age,city\nAarav,25,\n,31,Pune\nRohit,,Nagpur\n", "gaps.csv")
        analysis = dataset_analyzer.analyze_dataframe(df)
        missing = analysis["total_missing_cells"]
        assert missing == 3, f"expected 3 missing cells, got {missing}"
        return f"missing_cells={missing} rows={analysis['rows']}"
    check("A4 CSV with blank cells loads and reports missing values", a4)

    # A5 duplicate headers (pandas renames them a / a.1 unless caught first)
    expect_dataset_error("A5 duplicate column names are rejected",
                         b"name,age,name\nA,20,B\n", "dupes.csv", contains="duplicate")

    # A6 malformed CSV: a data row with more fields than the header
    expect_dataset_error("A6 inconsistent column count is rejected",
                         b"a,b\n1,2\n3,4,5\n", "ragged.csv", contains="column")

    # A6b empty / whitespace-only header cells
    def a6b():
        df = dataset_analyzer.load_csv_from_bytes(b" ,age,name\nx,20,b\n", "blankhdr.csv")
        assert all(str(c).strip() for c in df.columns), f"empty header survived: {list(df.columns)}"
        assert len(set(map(str, df.columns))) == len(df.columns), "columns must stay unique"
        return f"columns={list(df.columns)}"
    check("A6b whitespace-only header is given a safe unique name", a6b)

    # A6c binary content with a .csv name
    expect_dataset_error("A6c binary (non-text) upload is rejected",
                         b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00", "image.csv")

    # A6d single column
    expect_dataset_error("A6d single-column file is rejected",
                         b"name\nAarav\nDiya\n", "one_col.csv", contains="column")

    # A6e unsupported file type
    expect_dataset_error("A6e non-.csv file type is rejected",
                         b"name,age\nA,20\n", "book.xlsx", contains="csv")


# ---------------------------------------------------------------------------
# B. Prompt parsing / validation
# ---------------------------------------------------------------------------
def parse_ok(text: str):
    return requirement_parser.parse_requirements(text)


def expect_parse_rejected(name: str, text: str) -> bool:
    def _run():
        try:
            spec = parse_ok(text)
        except ValueError as exc:
            msg = str(exc)
            assert msg.strip(), "validation message must not be empty"
            return f"rejected: {msg[:80]}"
        cols = [c["name"] for c in (spec.columns or [])]
        raise AssertionError(f"prompt was accepted (columns={cols}) but should have been rejected")
    return check(name, _run)


def test_prompt_validation():
    print("\n== B. Prompt interpretation & invalid-input handling ==")

    expect_parse_rejected("B1 gibberish prompt (asdfghjkl qwerty xyz)", "asdfghjkl qwerty xyz")
    expect_parse_rejected("B2 gibberish prompt (blah blah blah make data thing)",
                          "blah blah blah make data thing")
    expect_parse_rejected("B3 empty prompt", "")
    expect_parse_rejected("B4 whitespace-only prompt", "    \n\t  ")

    # B5 meaningful but misspelled column names must still be understood
    def b5():
        s = parse_ok("give me 1000 rows wth name, age, mobile numbaer and email")
        sems = {c["name"]: c["semantic_type"] for c in s.columns}
        by_sem = {c["semantic_type"] for c in s.columns}
        assert s.rows == 1000, s.rows
        assert "indian_phone" in by_sem, f"mobile numbaer not understood: {sems}"
        assert "person_name" in by_sem and "age" in by_sem and "email" in by_sem, sems
        return f"rows={s.rows} columns={[(c['name'], c['semantic_type']) for c in s.columns]}"
    check("B5 spelling mistakes / list without 'with' still parsed", b5)

    # B6 unfamiliar but meaningful column names keep their meaning, no junk columns
    def b6():
        s = parse_ok("make a dataset with loyalty_tier and referral_code columns, 300 rows")
        names = [c["name"] for c in s.columns]
        assert s.rows == 300, s.rows
        assert len(names) == 2, names
        assert "Loyalty Tier" in names, names
        assert any(n.lower().replace(" ", "_") == "referral_code" for n in names), names
        assert not any(re.search(r"columns?$", n.lower()) for n in names), f"junk suffix: {names}"
        return f"rows={s.rows} columns={names}"
    check("B6 unfamiliar-but-meaningful column names are kept (no junk)", b6)

    # B6b the row-count clause must not become a column
    def b6b():
        s = parse_ok("I want a dataset with name, age, mobile number, email, city, and salary "
                     "columns, with 500 rows of realistic Indian data.")
        names = [c["name"] for c in s.columns]
        assert s.rows == 500, s.rows
        assert len(names) == 6, names
        assert not any(re.search(r"\d", n) for n in names), f"row count leaked into columns: {names}"
        assert s.country == "India", s.country
        return f"rows={s.rows} columns={names}"
    check("B6b row-count clause does not create a junk column", b6b)

    # B7 explicit row count
    def b7():
        s = parse_ok("Generate 750 records with name and age.")
        assert s.rows == 750, s.rows
        return f"rows={s.rows}"
    check("B7 requested row count is read from the prompt", b7)

    # B7b prompt with a category list and a numeric range
    def b7b():
        s = parse_ok("Create 1000 employee records with employee_id, department with values such "
                     "as HR, Finance, IT, age between 22 and 60 and salary.")
        names = [c["name"] for c in s.columns]
        assert s.rows == 1000, s.rows
        assert len(names) == 4, f"junk/missing columns: {names}"
        dept = next((c for c in s.columns if c["categories"]), None)
        assert dept and dept["categories"] == ["HR", "Finance", "IT"], dept
        age = next((c for c in s.columns if c["semantic_type"] == "age"), None)
        assert age and (age["min"], age["max"]) == (22, 60), age
        assert any(c["semantic_type"] == "customer_id" for c in s.columns), \
            [c["semantic_type"] for c in s.columns]
        return f"columns={names} categories={dept['categories']} age={age['min']}-{age['max']}"
    check("B7b category list + range + id column parsed", b7b)


# ---------------------------------------------------------------------------
# C. Generation accuracy
# ---------------------------------------------------------------------------
_PHONE_RE = re.compile(r"^[6-9]\d{9}$")
_PIN_RE = re.compile(r"^[1-9]\d{5}$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[a-zA-Z]{2,}$")
_FAKE_RUN = re.compile(r"(\d)\1{3,}|0123456789|1234567890|9876543210")


def _generate(text: str, rows: int, seed: int = 42):
    spec = parse_ok(text)
    spec.rows = rows
    spec.unique = False
    return spec, spec_synthesizer.generate_from_spec(spec, seed=seed)


def test_generation_accuracy():
    print("\n== C. Generation accuracy (schema, formats, ranges, consistency) ==")

    # C1 realistic Indian data: names, ages, mobiles, emails, DOB, pincode, city, salary
    text = ("Create 400 rows of realistic Indian data with name, age, date_of_birth, mobile "
            "number, email, city, state, pincode, address and salary. Age should be between "
            "21 and 60.")

    def c1():
        spec, res = _generate(text, 400)
        df = res["dataframe"]
        assert len(df) == 400, len(df)
        assert list(df.columns) == [c["name"] for c in spec.columns], list(df.columns)

        phone = df["Phone"].astype(str)
        phone_ok = float(phone.str.match(_PHONE_RE).mean()) * 100
        fake = int(phone.map(lambda v: bool(_FAKE_RUN.search(v)) or len(set(v)) <= 2).sum())
        email_ok = float(df["Email"].astype(str).str.match(_EMAIL_RE).mean()) * 100
        pin_ok = float(df["Pincode"].astype(str).str.match(_PIN_RE).mean()) * 100
        ages = pd.to_numeric(df["Age"], errors="coerce")
        age_ok = float(((ages >= 21) & (ages <= 60) & (ages == ages.round())).mean()) * 100
        dob = pd.to_datetime(df["Date of Birth"], errors="coerce")
        dob_ok = float(dob.notna().mean()) * 100
        # DOB must agree with the requested age (|age_from_dob - age| <= 1)
        implied = ((pd.Timestamp.today() - dob).dt.days / 365.25).round()
        age_dob_ok = float(((implied - ages).abs() <= 1).mean()) * 100
        names_ok = float(df["Name"].astype(str).str.match(r"^[A-Z][a-z]+ [A-Z][a-z]+$").mean()) * 100
        salary = pd.to_numeric(df["Salary"], errors="coerce")
        sal_ok = float(((salary >= 200000) & (salary <= 2500000)).mean()) * 100
        missing = int(df.isna().sum().sum())

        print(f"        phone_valid={phone_ok:.1f}% fake_phone={fake} email_valid={email_ok:.1f}% "
              f"pincode_valid={pin_ok:.1f}% age_in_range={age_ok:.1f}% dob_valid={dob_ok:.1f}% "
              f"dob~age={age_dob_ok:.1f}% names={names_ok:.1f}% salary_in_range={sal_ok:.1f}% "
              f"missing={missing} distinct_phones={phone.nunique()}")
        assert phone_ok == 100, f"{100 - phone_ok:.1f}% invalid phone numbers"
        assert fake == 0, f"{fake} obviously fake phone numbers"
        assert email_ok == 100 and pin_ok == 100, (email_ok, pin_ok)
        assert age_ok == 100, age_ok
        assert dob_ok == 100, dob_ok
        assert age_dob_ok >= 99, age_dob_ok
        assert names_ok >= 99, names_ok
        assert sal_ok == 100, sal_ok
        assert missing == 0, missing
        assert df.duplicated().sum() == 0, "duplicate rows generated"
        assert phone.nunique() == len(phone) or len(phone) > 3000, "phone numbers repeat"
        assert res["validation"]["passed"], [c for c in res["validation"]["checks"] if not c["passed"]]
        return f"rows={len(df)} quality_pass={res['validation']['passed']}"
    check("C1 realistic Indian data matches requested types/formats/ranges", c1)

    # C2 cross-column consistency when columns are NOT named Gender/State
    def c2():
        spec = requirement_parser.parse_requirements(
            "Create 300 rows with name, gender, state, city, pincode, address and email.")
        # simulate a user renaming columns in the schema editor
        rename = {"Gender": "sex", "State": "province"}
        for c in spec.columns:
            if c["name"] in rename:
                c["name"] = rename[c["name"]]
        spec.rows = 300
        df = spec_synthesizer.generate_from_spec(spec, seed=42)["dataframe"]

        states = df["province"].astype(str)
        cities = df["City"].astype(str)
        state_ok = float(sum(
            c in semantic_generators._STATE_CITIES.get(s, [])
            for s, c in zip(states, cities)) / len(df)) * 100

        pins = df["Pincode"].astype(str)
        pin_state_ok = float(sum(
            str(p)[:3] in semantic_generators._STATE_PIN_PREFIX.get(s, [str(p)[:3]])
            for s, p in zip(states, pins)) / len(df)) * 100

        emails = df["Email"].astype(str).str.split("@").str[0]
        email_name_ok = float(sum(
            nm.split()[0].lower() in em for nm, em in zip(df["Name"].astype(str), emails))
            / len(df)) * 100

        male = {"Male", "Female"}
        sub = df[df["sex"].isin(male)]
        gendered = [nm.split()[0] for nm in sub["Name"].astype(str)]
        gender_ok = float(sum(
            (g == "Male" and n in semantic_generators._MALE_FIRST) or
            (g == "Female" and n in semantic_generators._FEMALE_FIRST)
            for g, n in zip(sub["sex"], gendered)) / max(len(sub), 1)) * 100

        print(f"        state~city={state_ok:.1f}% pincode~state={pin_state_ok:.1f}% "
              f"email~name={email_name_ok:.1f}% name~gender={gender_ok:.1f}%")
        assert state_ok >= 99, state_ok
        assert pin_state_ok >= 99, pin_state_ok
        assert email_name_ok >= 99, email_name_ok
        assert gender_ok >= 95, gender_ok
        return f"rows={len(df)}"
    check("C2 relationships hold across renamed columns", c2)

    # C3 explicit row count + requested categories + ranges honoured
    def c3():
        spec = parse_ok("Generate 250 rows with name, gender with values such as Male and "
                        "Female, age between 30 and 45 and department with values such as HR "
                        "and Finance. Age should be between 30 and 45.")
        spec.rows = 250
        res = spec_synthesizer.generate_from_spec(spec, seed=42)
        df = res["dataframe"]
        age = pd.to_numeric(df["Age"], errors="coerce")
        cats_ok = set(df["Gender"]).issubset({"Male", "Female"}) and \
            set(df["Department"]).issubset({"HR", "Finance"})
        ages_ok = bool(((age >= 30) & (age <= 45)).all())
        print(f"        rows={len(df)} age_range=[{age.min()},{age.max()}] "
              f"gender={sorted(set(df['Gender']))} dept={sorted(set(df['Department']))} "
              f"qc_passed={res['validation']['passed']}")
        assert len(df) == 250, len(df)
        assert sorted(df.columns) == sorted(["Gender", "Department", "Name", "Age"]), \
            f"value fragments became columns: {list(df.columns)}"
        assert cats_ok and ages_ok, "requested categories/ranges not respected"
        assert res["validation"]["passed"], [c for c in res["validation"]["checks"] if not c["passed"]]
        return f"rows={len(df)} categories+ranges honoured"
    check("C3 requested row count, categories and ranges are honoured", c3)

    # C4 schema errors must be reported instead of silently generating
    def c4():
        spec = parse_ok("Create 100 rows with name and age.")
        spec.rows = 100
        spec.columns.append(dict(spec.columns[0]))  # duplicate column name
        try:
            spec_synthesizer.generate_from_spec(spec, seed=42)
        except ValueError as exc:
            assert "duplicate" in str(exc).lower(), str(exc)
            return f"rejected: {exc}"
        raise AssertionError("duplicate column names were silently accepted")
    check("C4 duplicate column names in the schema are rejected", c4)

    # C5 an impossible range ("between 65 and 18") is swapped + flagged when it
    # comes from a prompt, and rejected outright when it reaches generation
    # (i.e. was edited badly in the schema editor)
    def c5():
        spec = parse_ok("Create 100 rows with age between 65 and 18.")
        age = next(c for c in spec.columns if c["semantic_type"] == "age")
        assert (age["min"], age["max"]) == (18, 65), age
        assert any("swapped" in w for w in spec.warnings), spec.warnings
        spec.rows = 100
        res = spec_synthesizer.generate_from_spec(spec, seed=42)
        ages = pd.to_numeric(res["dataframe"]["Age"], errors="coerce")
        assert bool(((ages >= 18) & (ages <= 65)).all()), \
            f"out-of-range ages: {ages.min()}-{ages.max()}"
        return f"swapped to {age['min']}-{age['max']}, generated [{ages.min()},{ages.max()}]"
    check("C5 reversed range in a prompt is corrected, not mis-generated", c5)

    def c5b():
        spec = parse_ok("Create 100 rows with age between 18 and 65.")
        spec.rows = 100
        spec.columns[0]["min"], spec.columns[0]["max"] = 65, 18  # edited badly in the UI
        try:
            spec_synthesizer.generate_from_spec(spec, seed=42)
        except ValueError as exc:
            assert "minimum" in str(exc) and "maximum" in str(exc), str(exc)
            return f"rejected: {exc}"
        raise AssertionError("min > max from the schema editor was silently accepted")
    check("C5b min > max from the schema editor is rejected clearly", c5b)

    # C6 generation quality validator must catch invalid dates
    def c6():
        df = pd.DataFrame({"dob": ["2000-01-01", "not a date", "1999-13-45"],
                           "n": [1, 2, 3]})
        rep = generation_validator.validate_generation(
            df, column_specs=[{"name": "dob", "technical_type": "date",
                               "semantic_type": "date_of_birth"}], n_requested=3)
        bad = [c for c in rep["checks"] if c["check"] == "Format of 'dob'"]
        assert bad, f"date validity not checked: {rep['checks']}"
        assert bad[0]["passed"] is False, bad
        return bad[0]["detail"]
    check("C6 invalid date values are detected by the quality check", c6)

    # C7 a requested skew ("Most ... from Maharashtra" / "60% from Mumbai")
    def c7():
        spec = parse_ok("Create 500 rows with name, state, city and age. Most customers should "
                        "be from Maharashtra. 60% of customers should be from Mumbai.")
        spec.rows = 500
        df = spec_synthesizer.generate_from_spec(spec, seed=42)["dataframe"]
        mh = float((df["State"] == "Maharashtra").mean()) * 100
        coherent = all(
            c in semantic_generators._STATE_CITIES.get(s, [])
            for s, c in zip(df["State"].astype(str), df["City"].astype(str)))
        print(f"        Maharashtra={mh:.1f}% (asked ~60) distinct_states={df['State'].nunique()} "
              f"state~city_coherent={coherent}")
        assert 45 <= mh <= 75, f"skew not respected: {mh}%"
        assert df["State"].nunique() > 1, "a 'most' skew must not make every row identical"
        assert coherent, "the city column must stay consistent with the state column"
        return f"Maharashtra={mh:.1f}% distinct_states={df['State'].nunique()}"
    check("C7 requested state skew is respected (and not degenerate)", c7)

    def c7b():
        # without a state column the city skew itself must be applied
        spec = parse_ok("Create 500 rows with name, city and age. 60% of customers should be "
                        "from Mumbai.")
        spec.rows = 500
        df = spec_synthesizer.generate_from_spec(spec, seed=42)["dataframe"]
        mum = float((df["City"] == "Mumbai").mean()) * 100
        print(f"        Mumbai={mum:.1f}% (asked 60) distinct_cities={df['City'].nunique()}")
        assert 45 <= mum <= 75, f"city skew not respected: {mum}%"
        assert df["City"].nunique() > 1, "a 60% skew must not make every row identical"
        return f"Mumbai={mum:.1f}%"
    check("C7b requested city skew is applied to the city column", c7b)


# ---------------------------------------------------------------------------
# D. Regression of documented prompts
# ---------------------------------------------------------------------------
def test_regression():
    print("\n== D. Regression: documented example prompts ==")

    def d1():
        s = parse_ok(
            "I need a classification dataset with 5000 records for predicting customer churn. "
            "Include age, income, tenure, monthly charges and contract type. "
            "The target should be churn with approximately 20% positive cases.")
        assert s.rows == 5000, s.rows
        assert s.task == "classification", s.task
        assert (s.target or "").lower() == "churn", s.target
        assert s.class_distribution and s.class_distribution["positive_pct"] == 20, s.class_distribution
        assert len(s.columns) == 5, [c["name"] for c in s.columns]
        return f"rows={s.rows} task={s.task} target={s.target} columns={[c['name'] for c in s.columns]}"
    check("D1 churn prompt unchanged (rows/target/distribution/columns)", d1)

    def d2():
        s = parse_ok("Create 2000 unique Indian customer records with name, gender, age, address, "
                     "city, state, Indian phone number, email and salary. Age should be between "
                     "18 and 65.")
        names = [c["name"] for c in s.columns]
        assert s.rows == 2000 and s.country == "India" and s.unique, (s.rows, s.country, s.unique)
        assert names == ["Name", "Gender", "Age", "Address", "City", "State", "Phone",
                         "Email", "Salary"], names
        age = next(c for c in s.columns if c["name"] == "Age")
        assert (age["min"], age["max"]) == (18, 65), age
        return f"columns={names}"
    check("D2 Indian-records example prompt unchanged", d2)

    def d3():
        s = parse_ok("Generate 2000 employee records with employee name, department with values "
                     "such as HR, Finance, IT, Sales, age and salary.")
        dept = next(c for c in s.columns if c["categories"])
        assert dept["categories"] == ["HR", "Finance", "IT", "Sales"], dept
        assert s.rows == 2000
        return f"departments={dept['categories']}"
    check("D3 department-list example prompt unchanged", d3)

    def d4():
        s = parse_ok("Create a student dataset with name, age, gender and marks. "
                     "Marks should be between 0 and 100.")
        marks = next(c for c in s.columns if c["name"] == "Marks")
        assert (marks["min"], marks["max"]) == (0, 100), marks
        assert [c["name"] for c in s.columns] == ["Name", "Age", "Gender", "Marks"]
        assert any(q["field"] == "rows" for q in s.clarifications), "row count should be asked"
        return "marks 0-100, rows clarification asked"
    check("D4 student-dataset example prompt unchanged", d4)

    def d5():
        s = parse_ok("I want some data.")
        assert s.clarifications, "ambiguous-but-valid prompt must ask clarifications"
        return f"clarifications={[q['field'] for q in s.clarifications]}"
    check("D5 ambiguous-but-valid prompt still returns clarifications", d5)

    # D6 a normal Mode A style generation must still work end to end
    def d6():
        sample = Path(__file__).resolve().parent / "sample_data" / "telco_churn.csv"
        assert sample.exists(), "run scripts/make_sample_data.py first"
        df = dataset_analyzer.load_csv_from_bytes(sample.read_bytes(), sample.name)
        analysis = dataset_analyzer.analyze_dataframe(df)
        assert analysis["rows"] == len(df)
        return f"rows={analysis['rows']} cols={analysis['columns']} warnings={len(analysis['warnings'])}"
    check("D6 normal CSV upload path still works", d6)


# ---------------------------------------------------------------------------
# E. Mode A data quality through the REAL endpoints (in-process TestClient)
# ---------------------------------------------------------------------------
STATE: dict = {}   # shared across sections E -> F -> G (one in-process app)


def _client():
    if "client" not in STATE:
        from fastapi.testclient import TestClient
        from app.main import app
        STATE["client"] = TestClient(app)
    return STATE["client"]


def _upload_csv(client, df, filename) -> dict:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    r = client.post("/api/upload", files={
        "file": (filename, io.BytesIO(buf.getvalue().encode("utf-8")), "text/csv")})
    assert r.status_code == 200, f"upload failed {r.status_code}: {r.text[:200]}"
    return r.json()


def _generate_rows(client, upload_id, n_rows) -> dict:
    r = client.post(f"/api/upload/{upload_id}/generate",
                    json={"n_rows": n_rows, "synthesizer": "GaussianCopula"})
    assert r.status_code == 200, f"generate failed {r.status_code}: {r.text[:300]}"
    return r.json()


def _download_csv(client, path):
    r = client.get(path)
    assert r.status_code == 200, f"download failed {r.status_code}: {r.text[:200]}"
    cd = r.headers.get("content-disposition", "")
    m = re.search(r'filename="([^"]+)"', cd)
    return pd.read_csv(io.StringIO(r.content.decode("utf-8"))), (m.group(1) if m else "")


def test_mode_a_data_quality():
    print("\n== E. Mode A data quality via real endpoints (no server needed) ==")
    client = _client()
    rng = random.Random(3)
    n = 200

    # E1 — Age is a unique sequential counter, Address is unique per row:
    # SDV auto-classifies both as identifiers (the reported corruption:
    # 8-digit "ages" and "sdv-id-..." addresses), while the small name pool
    # used to repeat the same handful of names.
    def e1():
        df1 = pd.DataFrame({
            "Name": [f"{rng.choice(['John', 'Emily', 'Michael', 'Sarah', 'Anna'])} "
                     f"{rng.choice(['Smith', 'Davis', 'Johnson', 'Brown', 'Wilson'])}"
                     for _ in range(n)],
            "Age": list(range(22, 22 + n)),
            "Address": [f"{i + 1} Unique Lane, Block {i + 1}" for i in range(n)],
            "Employee ID": [f"EMP{i + 1:05d}" for i in range(n)],
        })
        up = _upload_csv(client, df1, "idlike.csv")
        _generate_rows(client, up["upload_id"], n)
        syn, dl_name = _download_csv(client, f"/api/upload/{up['upload_id']}/synthetic/download")

        # addresses: human-readable, never id tokens
        addr = syn["Address"].astype(str)
        sdv_tok = int(addr.str.contains("sdv-", case=False).sum())
        uuid_tok = int(addr.str.contains(r"[0-9a-f]{32}", regex=True).sum())
        multipart = float(addr.str.contains(", ").mean()) * 100
        with_digit = float(addr.str.contains(r"\d").mean()) * 100
        # age: within the SOURCE's own range, integer, not a counter, no garbage
        ages = pd.to_numeric(syn["Age"], errors="coerce")
        span = 221 - 22
        tol = max(span * 0.15, 2)
        in_range = float(((ages >= 22 - tol) & (ages <= 221 + tol)).mean()) * 100
        garbage = int((ages.abs() >= 10000).sum())
        seq_frac = float((ages.diff() == 1).mean())
        ints_ok = bool((ages == ages.round()).all())
        # names: the 25-combo source pool must not dominate 200 rows
        names = syn["Name"].astype(str)
        uniq_pct = names.nunique() / len(names) * 100
        top_freq = int(names.value_counts().iloc[0])
        # employee ids keep the source's EMP prefix and stay unique
        eids = syn["Employee ID"].astype(str)
        emp_ok = float(eids.str.match(r"^EMP\d+$").mean()) * 100
        eid_uniq = int(eids.nunique())

        print(f"        addr sdv-tokens={sdv_tok} uuid={uuid_tok} multipart={multipart:.0f}% "
              f"digit={with_digit:.0f}% | age range=[{ages.min():.0f},{ages.max():.0f}] "
              f"in_range={in_range:.1f}% garbage={garbage} seq={seq_frac:.3f} ints={ints_ok} "
              f"| names {names.nunique()}/{len(names)} unique top_freq={top_freq} "
              f"| emp_prefix={emp_ok:.0f}% uniq={eid_uniq} | dl={dl_name}")
        assert sdv_tok == 0 and uuid_tok == 0, f"{sdv_tok} sdv-id / {uuid_tok} uuid tokens in Address"
        assert multipart >= 95 and with_digit >= 95, (multipart, with_digit)
        assert in_range == 100, f"{100 - in_range:.1f}% ages outside the source range"
        assert garbage == 0, f"{garbage} ID-like garbage values (>=10000) in Age"
        assert seq_frac < 0.5, f"Age follows a +1 counter (seq={seq_frac:.2f})"
        assert ints_ok, "Age contains non-integer values"
        assert uniq_pct >= 90 and top_freq <= 5, f"name diversity too low: {uniq_pct:.1f}% unique, top x{top_freq}"
        assert emp_ok == 100 and eid_uniq == len(eids), (emp_ok, eid_uniq)
        return f"age=[{ages.min():.0f},{ages.max():.0f}] names={names.nunique()}/{len(names)}"
    check("E1 unique Age/Address stay human-readable, names diverse", e1)

    # E2 — realistic Indian employee dataset: cross-column consistency
    first = ["Rahul", "Priya", "Aarav", "Ananya", "Vikram", "Sneha", "Karan", "Isha", "Rohit", "Meera"]
    last = ["Sharma", "Patel", "Singh", "Gupta", "Reddy", "Iyer"]
    cities = {"Mumbai": "Maharashtra", "Pune": "Maharashtra", "Delhi": "Delhi"}
    n2, rows_n = 250, 300
    rows = []
    for i in range(n2):
        city = rng.choice(list(cities))
        age = rng.randint(22, 58)
        dob_year = 2026 - age
        hire_year = min(2024, dob_year + 20 + rng.randint(0, 15))
        rows.append({
            "Full Name": f"{rng.choice(first)} {rng.choice(last)}",
            "Gender": rng.choice(["Male", "Female"]),
            "Department": rng.choice(["HR", "IT", "Finance"]),
            "Job Title": rng.choice(["Executive", "Engineer", "Accountant"]),
            "Salary": rng.randint(3, 25) * 100000,
            "Date of Birth": f"{dob_year}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
            "Age": age,
            "Hire Date": f"{hire_year}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
            "Employee ID": f"EMP{i + 1:05d}",
            "Address": f"{rng.randint(1, 999)} Oak Street, Locality {i}",
            "Mobile Number": f"98{10000000 + i}",
            "City": city,
            "State": cities[city],
            "Email": f"user{i}@example.com",
        })
    df2 = pd.DataFrame(rows)

    def e2():
        up = _upload_csv(client, df2, "employees.csv")
        STATE["employee_upload"] = up
        gen = _generate_rows(client, up["upload_id"], rows_n)
        syn, _ = _download_csv(client, f"/api/upload/{up['upload_id']}/synthetic/download")
        STATE["employee_synth"] = syn
        STATE["employee_rows"] = len(syn)

        ages = pd.to_numeric(syn["Age"], errors="coerce")
        dob = pd.to_datetime(syn["Date of Birth"], errors="coerce")
        hire = pd.to_datetime(syn["Hire Date"], errors="coerce")
        gap = ((pd.Timestamp.today() - dob).dt.days / 365.25 - ages).abs()
        dob_age_ok = float((gap <= 1).mean()) * 100
        hire_viol = int((hire < dob).sum())

        city_state_ok = float(sum(
            cities.get(c) == s for c, s in
            zip(syn["City"].astype(str), syn["State"].astype(str))) / len(syn)) * 100

        mob = syn["Mobile Number"].astype(str).str.replace(r"\.0$", "", regex=True)
        mob_ok = float(mob.str.match(r"^[6-9]\d{9}$").mean()) * 100

        addr = syn["Address"].astype(str)
        addr_ok = float(sum(
            c in a and s in a for c, s, a in
            zip(syn["City"].astype(str), syn["State"].astype(str), addr)) / len(syn)) * 100

        names = syn["Full Name"].astype(str)
        email_ok = float(sum(
            nm.split()[0].lower() in e for nm, e in
            zip(names, syn["Email"].astype(str))) / len(syn)) * 100

        # the misdetected "Job Title" must stay categories, never become names
        titles = set(syn["Job Title"].astype(str))
        src_titles = set(df2["Job Title"].astype(str))
        title_ok = titles.issubset(src_titles)

        in_range = float(((ages >= 22 - 5.4) & (ages <= 58 + 5.4)).mean()) * 100
        uniq_pct = names.nunique() / len(names) * 100
        missing = int(syn.isna().sum().sum())

        print(f"        dob~age={dob_age_ok:.1f}% hire<dob={hire_viol} city~state={city_state_ok:.1f}% "
              f"mobile={mob_ok:.1f}% addr~place={addr_ok:.1f}% email~name={email_ok:.1f}% "
              f"titles_subset={title_ok} age_range={in_range:.1f}% names={uniq_pct:.0f}% unique "
              f"missing={missing}")
        assert dob_age_ok >= 99, dob_age_ok
        assert hire_viol == 0, f"{hire_viol} hire dates before date of birth"
        assert city_state_ok >= 99, city_state_ok
        assert mob_ok == 100, mob_ok
        assert addr_ok >= 99, addr_ok
        assert email_ok >= 99, email_ok
        assert title_ok, f"Job Title invented values: {sorted(titles - src_titles)}"
        assert in_range == 100, in_range
        assert uniq_pct >= 80, uniq_pct
        assert missing == 0, f"{missing} missing cells"
        assert len(syn) == rows_n, len(syn)
        assert list(syn.columns) == list(df2.columns), "columns changed"
        assert gen.get("approach"), "no generation approach reported"
        return f"rows={len(syn)} approach={gen.get('approach')}"
    check("E2 employee dataset keeps DOB/hire/city/name/email relationships", e2)


# ---------------------------------------------------------------------------
# F. PDF report generation
# ---------------------------------------------------------------------------
def _report_payload(quantiles, with_mean_keys=True) -> dict:
    """Minimal stored-evaluation payload shaped like evaluation_engine's."""
    stats = {"quantiles": quantiles}
    if with_mean_keys:
        stats["mean"] = {"real": None, "synthetic": None}
        stats["std"] = {"real": None, "synthetic": None}
    return {
        "name": "F fixture", "mode": "upload", "created_at": "2026-10-23 00:46",
        "dataset_filename": "fixture.csv", "rows_real": 10, "columns": 2,
        "rows_requested": 10, "rows_synthetic": 10,
        "fit_seconds": 0.3, "sample_seconds": 0.1,
        "analysis": {"rows": 10, "columns": 2, "total_missing_cells": 0,
                     "duplicate_rows": 0, "constant_columns": [], "id_like_columns": [],
                     "dtypes_summary": {}, "warnings": []},
        "detection": {}, "ml_utility": None, "explanation": {}, "warnings": [],
        "evaluation": {
            "data_nova_score": {"value": None, "components": {"statistical": None}},
            "column_overlap": {"shared": ["Age"], "only_in_real": [], "only_in_synthetic": []},
            "row_counts": {"real": 10, "synthetic": 10},
            "column_results": [
                {"column": "Age", "similarity": None, "stats": stats},
                {"column": "Dept", "similarity": 0.5,
                 "proportions": [{"category": "HR", "real_share": None, "synthetic_share": 0.5}]},
            ],
            "correlation_comparison": {"mean_abs_difference": None,
                                        "real_matrix": [[0.1, None], [None, 1.0]],
                                        "synthetic_matrix": [[0.1, 0.2], [0.2, 1.0]],
                                        "columns": ["Age", "Dept"]},
            "target_comparison": {"kind": "classification", "similarity": None},
        },
        "synth_preview": {"file": "z.csv", "rows": None},
    }


def _assert_valid_pdf(path: Path, min_bytes: int) -> int:
    data = path.read_bytes()
    assert data[:5] == b"%PDF-", f"not a PDF: {data[:16]!r}"
    assert len(data) >= min_bytes, f"suspiciously small PDF: {len(data)} bytes"
    return len(data)


def _preclean_fixture_files() -> None:
    """Remove residue of a previous crashed run so names stay deterministic."""
    for name in ("indian-customer-dataset.csv", "indian-customer-dataset_2.csv",
                 "indian-customer-messy.csv", "legacy-20240101-synthetic-1030.csv",
                 "employee-records-dataset.csv", "employee-records-dataset_2.csv"):
        try:
            (config.DATA_DIR / "synthetic" / name).unlink(missing_ok=True)
        except OSError:
            pass
    try:
        (config.REPORT_DIR / "indian-customer-report.pdf").unlink(missing_ok=True)
    except OSError:
        pass


def test_report_generation():
    print("\n== F. PDF report generation ==")
    _preclean_fixture_files()

    # F1 — the reported crash: an evaluation whose quantiles are ALL None
    # (empty numeric column after dropping missing values) used to raise
    # `TypeError: unsupported operand type(s) for +: 'int' and 'NoneType'`.
    def f1():
        import tempfile
        out = Path(tempfile.mkdtemp()) / "all_none.pdf"
        payload = _report_payload({"real": [None] * 5, "synthetic": [None] * 5})
        build_report(payload, out)
        size = _assert_valid_pdf(out, 100_000)
        try:
            out.unlink()
        except OSError:
            pass
        return f"size={size} bytes"
    check("F1 all-None quantiles report generates (was: int + None crash)", f1)

    # F2 — partial-None quantiles and stats entries without mean/std keys
    def f2():
        import tempfile
        out = Path(tempfile.mkdtemp()) / "partial_none.pdf"
        payload = _report_payload(
            {"real": [1, None, 3, None, 5], "synthetic": [2, 4, None, 8, 10]},
            with_mean_keys=False)   # column stats lacking mean/std entirely
        build_report(payload, out)
        size = _assert_valid_pdf(out, 100_000)
        try:
            out.unlink()
        except OSError:
            pass
        return f"size={size} bytes"
    check("F2 partial-None / missing-key stats render without error", f2)

    # F3 — real end-to-end: evaluate -> save project -> report -> download
    def f3():
        client = _client()
        up = STATE["employee_upload"]
        uid = up["upload_id"]

        ev = client.post("/api/evaluate", json={"upload_id": uid})
        assert ev.status_code == 200, f"evaluate {ev.status_code}: {ev.text[:200]}"
        evaluation = ev.json()

        save = client.post("/api/projects/save", json={
            "name": "Indian Customer Synthetic Dataset",
            "mode": "upload",
            "upload_id": uid,
            "dataset_filename": up["filename"],
            "rows_real": up["analysis"]["rows"],
            "columns": up["analysis"]["columns"],
            "rows_synthetic": STATE["employee_rows"],
            "synthesizer": "GaussianCopula",
            "analysis": up["analysis"],
            "detection": up["detection"],
            "evaluation": evaluation,
        })
        assert save.status_code == 200, f"save {save.status_code}: {save.text[:200]}"
        pid1 = save.json()["project_id"]
        STATE["pid1"] = pid1

        rep = client.post(f"/api/projects/{pid1}/report")
        assert rep.status_code == 200, f"report {rep.status_code}: {rep.text[:300]}"
        assert rep.json()["filename"] == "indian-customer-report.pdf", rep.json()
        STATE["report_filename"] = rep.json()["filename"]

        dl = client.get(f"/api/projects/{pid1}/report/download")
        assert dl.status_code == 200, f"report download {dl.status_code}"
        assert dl.content[:5] == b"%PDF-", dl.content[:16]
        assert len(dl.content) > 100_000, len(dl.content)

        # the report must contain the project's own information
        text = extract_pdf_text(Path(rep.json()["report_path"]))
        norm = re.sub(r"\s+", " ", text.encode("ascii", "ignore").decode())
        needles = ["Indian Customer Synthetic Dataset", "PROJECT REPORT",
                   "Expanded Dataset (Mode A)", "DataNova Score", "employees.csv"]
        missing = [x for x in needles if x not in norm]
        assert not missing, f"report is missing {missing} (chars={len(text)})"
        return f"pdf={len(dl.content)} bytes checks={len(needles)}/{len(needles)}"
    check("F3 real project report is valid PDF with correct project info", f3)


# ---------------------------------------------------------------------------
# G. Download file names: short, meaningful, collision-safe
# ---------------------------------------------------------------------------
def test_filenames():
    print("\n== G. Download file names (short, meaningful, collision-safe) ==")
    client = _client()
    up = STATE["employee_upload"]
    uid = up["upload_id"]

    # G1 session download: no date/time stamps, short subject stem
    def g1():
        syn, name = _download_csv(client, f"/api/upload/{uid}/synthetic/download")
        assert name == "employees-dataset.csv", name
        assert len(name) <= 40 and not re.search(r"\d{8}", name), name
        assert len(syn) == STATE["employee_rows"], len(syn)
        return name
    check("G1 session download name is short and stamp-free", g1)

    # G2 project save names the CSV after the project's subject
    # ("Employee Records Dataset" -> employee-records-dataset.csv;
    #  the "Indian Customer..." stem belongs to the F3 project)
    def g2():
        save = client.post("/api/projects/save", json={
            "name": "Employee Records Dataset", "mode": "upload",
            "upload_id": uid, "dataset_filename": up["filename"],
            "rows_real": up["analysis"]["rows"], "columns": up["analysis"]["columns"],
        })
        assert save.status_code == 200, save.text[:200]
        j = save.json()
        assert j["synthetic_file"] == "employee-records-dataset.csv", j
        assert (config.DATA_DIR / "synthetic" / j["synthetic_file"]).exists(), j
        STATE["pid1b"] = j["project_id"]
        return j["synthetic_file"]
    check("G2 project CSV is named after the subject (not a date-time slug)", g2)

    # G3 a second project with the same subject gets a collision suffix
    def g3():
        save = client.post("/api/projects/save", json={
            "name": "Employee Records Dataset Copy", "mode": "upload",
            "upload_id": uid, "dataset_filename": up["filename"],
        })
        assert save.status_code == 200, save.text[:200]
        j = save.json()
        assert j["synthetic_file"] == "employee-records-dataset_2.csv", j
        STATE["pid2"] = j["project_id"]
        return j["synthetic_file"]
    check("G3 filename collision gets a _2 suffix (no overwrite)", g3)

    # G4 report regeneration overwrites the project's OWN file
    def g4():
        pid1 = STATE["pid1"]
        first = client.post(f"/api/projects/{pid1}/report")
        second = client.post(f"/api/projects/{pid1}/report")
        assert first.status_code == 200 and second.status_code == 200, \
            (first.status_code, second.status_code)
        fn2 = second.json()["filename"]
        assert fn2 == "indian-customer-report.pdf", f"own report renamed to {fn2}"
        assert not Path(second.json()["report_path"]).name.endswith("_2.pdf"), fn2
        return fn2
    check("G4 report regeneration keeps the same file (own overwrite)", g4)

    # G5 messy copy keeps the "messy.csv" convention
    def g5():
        pid1 = STATE["pid1"]
        r = client.post(f"/api/projects/{pid1}/messy", json={"missing_values": 5.0})
        assert r.status_code == 200, f"messy {r.status_code}: {r.text[:200]}"
        assert r.json()["file"] == "indian-customer-messy.csv", r.json()["file"]
        syn, name = _download_csv(client, f"/api/projects/{pid1}/messy/download")
        assert name == "indian-customer-messy.csv", name
        return name
    check("G5 messy copy named '<subject>-messy.csv'", g5)

    # G6 session messy download: no time stamps either
    def g6():
        r = client.post(f"/api/upload/{uid}/messy", json={"missing_values": 5.0})
        assert r.status_code == 200, f"session messy {r.status_code}: {r.text[:200]}"
        syn, name = _download_csv(client, f"/api/upload/{uid}/messy/download")
        assert name == "employees-messy.csv", name
        assert not re.search(r"\d{4}\.csv$", name), name
        return name
    check("G6 session messy download name is short and stamp-free", g6)

    # G7 a project saved BEFORE the rename still downloads (legacy on-disk name)
    def g7():
        pid2 = STATE["pid2"]
        legacy = "legacy-20240101-synthetic-1030.csv"
        legacy_path = config.DATA_DIR / "synthetic" / legacy
        legacy_path.parent.mkdir(exist_ok=True)
        legacy_path.write_text("col_a,col_b\n1,2\n3,4\n", encoding="utf-8")
        database.update_project(pid2, {"synth_preview": json.dumps({"file": legacy, "rows": 2})})
        syn, name = _download_csv(client, f"/api/projects/{pid2}/synthetic/download")
        assert list(syn.columns) == ["col_a", "col_b"], list(syn.columns)
        assert legacy not in name, f"legacy long name leaked into download: {name}"
        assert name == "employee-records-dataset.csv", name
        return f"{name} (stored as {legacy})"
    check("G7 legacy stored file still downloads under a short name", g7)

    # --- best-effort cleanup of everything this run created ---
    for pid in (STATE.pop("pid1", None), STATE.pop("pid1b", None), STATE.pop("pid2", None)):
        if pid:
            try:
                client.delete(f"/api/projects/{pid}")
            except Exception:
                pass
    _preclean_fixture_files()


def main():
    print("DataNova accuracy & input-handling test")
    print("=" * 78)
    test_csv_uploads()
    test_prompt_validation()
    test_generation_accuracy()
    test_regression()
    test_mode_a_data_quality()
    test_report_generation()
    test_filenames()

    print("\n" + "=" * 78)
    for mark, name in results:
        print(f"{mark}  {name}")
    fails = [n for m, n in results if m == FAIL]
    print(f"\n{len(results) - len(fails)}/{len(results)} checks passed")
    if fails:
        print("FAILED: " + "; ".join(fails))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

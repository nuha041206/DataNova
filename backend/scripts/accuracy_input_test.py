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

Run:  python -X utf8 scripts/accuracy_input_test.py
Every check prints PASS/FAIL with its measured value; a summary follows.
"""
from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from app.services import (
    dataset_analyzer,
    generation_validator,
    requirement_parser,
    semantic_generators,
    spec_synthesizer,
)
from app.services.dataset_analyzer import DatasetError

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


def main():
    print("DataNova accuracy & input-handling test")
    print("=" * 78)
    test_csv_uploads()
    test_prompt_validation()
    test_generation_accuracy()
    test_regression()

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

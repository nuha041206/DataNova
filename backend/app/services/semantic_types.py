"""
DataNova semantic type detection.

Distinguishes TECHNICAL datatype (integer / float / string / boolean / date /
datetime / categorical — what the data IS in memory) from SEMANTIC datatype
(person_name / phone_number / email / indian_pincode / age / salary / ... —
what the data MEANS).

Examples this module must resolve correctly:
    "9876543210"   -> technical: string    semantic: phone_number
    "Rahul Sharma" -> technical: string    semantic: person_name
    "Male"         -> technical: string    semantic: gender
    "400001"       -> technical: string    semantic: indian_pincode
    34             -> technical: integer   semantic: age
    "24 MG Road"   -> technical: string    semantic: address

Detection combines name hints, value-pattern regexes and cardinality rules.
It is a transparent rules engine — no ML model is claimed.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Canonical vocabulary (shared with the frontend dropdowns)
# ---------------------------------------------------------------------------

TECHNICAL_TYPES = [
    "integer", "float", "string", "boolean", "date", "datetime", "categorical",
]

SEMANTIC_TYPES = [
    "generic_text", "person_name", "gender", "address", "city", "state",
    "country", "indian_phone", "email", "indian_pincode", "date_of_birth",
    "datetime", "date", "age", "customer_id", "generic_id", "product_name",
    "company_name", "salary", "currency", "percentage", "category", "integer",
    "float", "boolean", "custom",
]

# Human labels used by UI + reports
SEMANTIC_LABELS: Dict[str, str] = {
    "generic_text": "Generic Text",
    "person_name": "Person Name",
    "gender": "Gender",
    "address": "Address",
    "city": "City",
    "state": "State",
    "country": "Country",
    "indian_phone": "Indian Phone Number",
    "email": "Email",
    "indian_pincode": "Indian PIN Code",
    "date_of_birth": "Date of Birth",
    "datetime": "Datetime",
    "date": "Date",
    "age": "Age",
    "customer_id": "Customer ID",
    "generic_id": "Generic ID",
    "product_name": "Product Name",
    "company_name": "Company Name",
    "salary": "Salary",
    "currency": "Currency",
    "percentage": "Percentage",
    "category": "Category",
    "integer": "Integer",
    "float": "Float / Decimal",
    "boolean": "Boolean",
    "custom": "Custom",
}


def technical_type_of(series: pd.Series) -> str:
    """Technical datatype from the pandas dtype (what the data IS)."""
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_float_dtype(series):
        return "float"
    # object columns: probe the values
    sample = series.dropna().astype(str).head(200)
    if sample.empty:
        return "string"
    numeric = pd.to_numeric(sample.str.replace(",", "", regex=False), errors="coerce")
    if numeric.notna().mean() >= 0.95:
        # stored as text but numeric-looking (e.g. pin codes, phones WITHOUT
        # leading zeros/plus). Technical type stays as the file shows it.
        vals = numeric.dropna()
        if (vals == vals.astype(int)).all():
            return "integer" if series.dropna().map(lambda v: not str(v).startswith("0")).mean() > 0.9 else "string"
        return "float"
    try:
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        if parsed.notna().mean() >= 0.9:
            return "datetime"
    except Exception:
        pass
    low_card = series.nunique(dropna=True) <= max(20, len(series) * 0.05)
    if low_card:
        return "categorical"
    return "string"


# ---------------------------------------------------------------------------
# Value-pattern detectors (independent of column names)
# ---------------------------------------------------------------------------

_PHONE_RE = re.compile(r"^(?:\+?91[\s-]?)?[6-9]\d{9}$")          # Indian mobile
_PIN_RE = re.compile(r"^[1-9]\d{5}$")                             # Indian PIN
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[a-zA-Z]{2,}$")
_COORD_RE = re.compile(r"^-?\d{1,3}\.\d{4,},?\s*-?\d{1,3}\.\d{4,}$")

_INDIA_CITIES = {
    "mumbai", "delhi", "new delhi", "bengaluru", "bangalore", "hyderabad",
    "ahmedabad", "chennai", "kolkata", "surat", "pune", "jaipur", "lucknow",
    "kanpur", "nagpur", "indore", "thane", "bhopal", "visakhapatnam",
    "patna", "vadodara", "ghaziabad", "ludhiana", "nashik", "agra", "faridabad",
    "meerut", "rajkot", "varanasi", "srinagar", "amritsar", "navi mumbai",
    "allahabad", "ranchi", "howrah", "coimbatore", "jabalpur", "gwalior",
    "vijayawada", "jodhpur", "madurai", "raipur", "kochi", "chandigarh",
    "guwahati", "solapur", "hubli", "mysore", "thrissur", "kozhikode",
    "noida", "gurgaon", "gurugram", "thiruvananthapuram", "mangalore",
}

_INDIA_STATES = {
    "maharashtra", "gujarat", "karnataka", "tamil nadu", "delhi", "kerala",
    "west bengal", "uttar pradesh", "rajasthan", "telangana", "andhra pradesh",
    "madhya pradesh", "bihar", "punjab", "haryana", "odisha", "assam",
    "jharkhand", "chhattisgarh", "uttarakhand", "himachal pradesh", "goa",
    "tripura", "manipur", "meghalaya", "nagaland", "arunachal pradesh",
    "mizoram", "sikkim", "jammu and kashmir", "ladakh", "puducherry",
    "chandigarh", "andaman and nicobar islands", "dadra and nagar haveli",
}

_COUNTRY_NAMES = {
    "india", "united states", "usa", "united kingdom", "uk", "canada",
    "australia", "germany", "france", "japan", "china", "brazil", "south africa",
    "singapore", "uae", "united arab emirates", "netherlands", "spain", "italy",
}

_NAME_PARTICLES = {
    "kumar", "sharma", "patel", "singh", "gupta", "reddy", "rao", "mehta",
    "joshi", "desai", "iyer", "nair", "verma", "malhotra", "kapoor", "shah",
    "das", "banerjee", "mukherjee", "chatterjee", "iyengar", "menon", "pillai",
}


def _value_semantics(series: pd.Series) -> List[Tuple[str, float]]:
    """Score value patterns on a sample. Higher-confidence first."""
    sample = series.dropna().astype(str).head(300)
    if sample.empty:
        return []
    n = len(sample)
    scores: List[Tuple[str, float]] = []

    phone_hits = sample.str.match(_PHONE_RE).mean()
    pin_hits = sample.str.match(_PIN_RE).mean()
    email_hits = sample.str.match(_EMAIL_RE).mean()
    coord_hits = sample.str.match(_COORD_RE).mean()

    # NOTE: any 6-digit integer matches the PIN regex, so the PIN pattern is
    # only meaningful for TEXT columns (real pincodes are usually strings with
    # possible leading zeros). Numeric columns are handled by name hints.
    is_numeric_column = pd.api.types.is_numeric_dtype(series)

    if phone_hits >= 0.9 and not is_numeric_column:
        scores.append(("indian_phone", phone_hits))
    if email_hits >= 0.9:
        scores.append(("email", email_hits))
    if pin_hits >= 0.9 and not is_numeric_column:
        scores.append(("indian_pincode", pin_hits))
    if coord_hits >= 0.9:
        scores.append(("address", coord_hits))
    if phone_hits >= 0.9 and is_numeric_column:
        # A 10-digit integer column in the mobile range is still likely a
        # phone column stored as numbers (leading zero lost).
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if len(numeric) and numeric.min() >= 6_000_000_000 and numeric.max() <= 9_999_999_999:
            scores.append(("indian_phone", 0.75))

    low = sample.str.strip().str.lower()
    if low.isin(_INDIA_STATES).mean() >= 0.8:
        scores.append(("state", 0.95))
    if low.isin(_INDIA_CITIES).mean() >= 0.8:
        scores.append(("city", 0.9))
    if low.isin(_COUNTRY_NAMES).mean() >= 0.8:
        scores.append(("country", 0.9))
    if low.isin({"male", "female", "other", "m", "f", "non-binary"}).mean() >= 0.9:
        scores.append(("gender", 0.95))

    # Person names: 1-3 alphabetic words, each capitalised, some known surname
    name_like = sample.str.match(r"^[A-Z][a-z]+(?: [A-Z][a-z]+){0,2}$")
    surname_hits = low.str.contains("|".join(sorted(_NAME_PARTICLES)), regex=True).mean()
    if name_like.mean() >= 0.8:
        conf = 0.8 + min(0.15, surname_hits)
        scores.append(("person_name", conf))
    return scores


# ---------------------------------------------------------------------------
# Name-hint table (column name -> semantic type)
# ---------------------------------------------------------------------------

_NAME_HINTS: List[Tuple[Tuple[str, ...], str]] = [
    (("phone", "mobile", "contact_no", "contact number", "whatsapp", "msisdn"), "indian_phone"),
    (("pincode", "pin_code", "pin code", "zip", "zipcode", "postal"), "indian_pincode"),
    (("email", "e-mail", "mail_id"), "email"),
    (("full_name", "fullname", "customer_name", "employee_name", "student_name",
      "person_name", "first_name", "last_name", "surname", "name"), "person_name"),
    (("gender", "sex"), "gender"),
    (("address", "addr", "street", "locality", "residence"), "address"),
    (("city", "town", "district"), "city"),
    (("state", "province"), "state"),
    (("country", "nation"), "country"),
    (("dob", "date_of_birth", "birth"), "date_of_birth"),
    (("age",), "age"),
    (("salary", "income", "wage", "ctc", "package", "annum"), "salary"),
    (("price", "cost", "amount", "revenue", "fee", "charge", "expense", "premium", "budget", "spend", "bill"), "currency"),
    (("percentage", "percent", "pct", "ratio", "share", "rate"), "percentage"),
    (("customer_id", "cust_id", "user_id", "employee_id", "student_id", "account_no",
      "account_number", "card_no", "policy_no", "order_id", "invoice_no"), "customer_id"),
    (("product", "item", "sku", "commodity"), "product_name"),
    (("company", "organisation", "organization", "employer", "firm", "vendor", "brand"), "company_name"),
]


def _name_hint(name: str) -> Optional[str]:
    lname = name.lower().strip()
    if not lname:
        return None
    for keys, semantic in _NAME_HINTS:
        for k in keys:
            if k == lname or k in lname:
                return semantic
    return None


def detect_semantic_type(series: pd.Series) -> Dict[str, object]:
    """
    Full detection for one column.

    Returns:
        {
          "technical_type": "integer|float|string|boolean|date|datetime|categorical",
          "semantic_type": one of SEMANTIC_TYPES,
          "semantic_confidence": 0-1,
          "detection_reason": short human explanation,
          "value_sample": a few real values (for the UI),
        }
    """
    name = str(series.name)
    tech = technical_type_of(series)
    non_null = series.dropna()
    unique_ratio = (non_null.nunique() / len(non_null)) if len(non_null) else 0.0

    semantic = "generic_text"
    confidence = 0.4
    reason = "No stronger signal found; treated as text."

    # 1) boolean / datetime columns are unambiguous
    if tech == "boolean":
        return _result("boolean", "boolean", 1.0, "Column contains true/false values.", series)
    if tech == "datetime":
        dob_hint = _name_hint(name) == "date_of_birth"
        return _result("date_of_birth" if dob_hint else "datetime", "datetime",
                       0.9 if not dob_hint else 0.95,
                       "Parsed as dates" + ("; name suggests date of birth" if dob_hint else "."),
                       series)

    # 2) value patterns (strongest signal for strings)
    for cand, conf in _value_semantics(series):
        if conf >= confidence:
            semantic, confidence = cand, conf
            reason = f"Value patterns match {cand.replace('_', ' ')} ({conf:.0%} of sampled values)."

    # 3) name hints (weights depend on technical type)
    hint = _name_hint(name)
    if hint:
        hint_conf = 0.85 if tech in ("string", "categorical") else 0.8
        if hint == "indian_pincode" and tech == "integer":
            hint_conf = 0.75
        if hint == "indian_phone" and tech == "integer":
            hint_conf = 0.7  # leading zeros lost, but still a phone column
        if hint in ("age",) and tech not in ("integer", "float"):
            hint_conf = 0.3
        if hint == "indian_phone" and tech not in ("string", "integer"):
            hint_conf = 0.2
        if hint_conf > confidence:
            semantic, confidence = hint, hint_conf
            reason = f"Column name suggests {SEMANTIC_LABELS.get(hint, hint)}."

    # 4) numbers: sanity-guard pattern matches, then classify
    if tech in ("integer", "float"):
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if len(numeric):
            mn, mx = float(numeric.min()), float(numeric.max())
            # Guard: 6-digit PIN matches are only trusted for TEXT columns or
            # pincode-named columns; a numeric column with wide-ranging values
            # is a number, not a pincode (e.g. Salary 200000-2500000).
            if semantic == "indian_pincode" and hint != "indian_pincode" and \
                    (mx > 999999 or mn < 100000 or tech == "integer" and mx - mn > 900000):
                semantic, confidence = "generic_text", 0.4
                reason = "6-digit pattern match rejected: numeric values outside PIN range."
            if semantic == "indian_phone" and hint != "indian_phone" and tech == "integer" and \
                    not (6000000000 <= mn and mx <= 9999999999):
                semantic, confidence = "generic_text", 0.4
                reason = "10-digit pattern match rejected: values outside mobile range."

    if tech in ("integer", "float") and semantic in ("generic_text",):
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if len(numeric):
            mn, mx = float(numeric.min()), float(numeric.max())
            is_integer_like = tech == "integer" or (numeric == numeric.round()).all()
            # Guard: 6-digit-looking PIN matches are only trusted for TEXT
            # columns; a real integer column with wide-ranging values is a
            # number, not a pincode (e.g. Salary 200000-2500000).
            if semantic == "indian_pincode" and tech == "integer" and (mx > 999999 or mn < 100000):
                semantic = "generic_text"
                confidence = 0.4
                reason = "6-digit pattern match rejected: integer values outside PIN range."
            if semantic == "indian_phone" and tech == "integer":
                # 10-digit integers could be phones; keep only if in range
                if not (6000000000 <= mn <= 9999999999 and mx <= 9999999999):
                    semantic = "generic_text"
                    confidence = 0.4
                    reason = "10-digit pattern match rejected: values outside mobile range."
        if tech in ("integer", "float") and semantic in ("generic_text",):
            mn, mx = (float(numeric.min()), float(numeric.max())) if len(numeric) else (0.0, 0.0)
            is_integer_like = tech == "integer" or (len(numeric) and (numeric == numeric.round()).all())
            if hint is None and 0 <= mn and mx <= 110 and is_integer_like and 15 <= (numeric.mean() if len(numeric) else 0) <= 70:
                semantic, confidence = "age", 0.7
                reason = "Integer values in a plausible age range (15-70)."
            elif unique_ratio > 0.95 and is_integer_like and mn >= 0 and str(name).lower() in _id_names():
                semantic, confidence = "generic_id", 0.8
                reason = "Nearly-unique integers with an ID-like name."
            else:
                semantic, confidence = ("integer" if tech == "integer" else "float"), 0.8
                reason = f"Numeric column ({tech}); no special semantic detected."
        elif tech in ("integer", "float"):
            semantic, confidence = tech, 0.5

    # 5) categoricals without a specific semantic
    if semantic in ("generic_text",) and tech == "categorical":
        semantic, confidence = "category", 0.7
        reason = f"Low-cardinality text ({int(non_null.nunique())} distinct values); treated as categories."

    # 6) unique strings are IDs
    if semantic == "generic_text" and tech == "string" and unique_ratio > 0.95:
        semantic, confidence = "generic_id", 0.6
        reason = "Nearly every value is unique; treated as an identifier."

    return _result(tech, semantic, confidence, reason, series)


def _id_names() -> set:
    return {"id", "no", "num", "number", "index", "sr_no", "s_no", "sl_no", "row_id"}


def _result(tech: str, semantic: str, conf: float, reason: str, series: pd.Series) -> Dict[str, object]:
    sample = series.dropna().head(5)
    return {
        "technical_type": tech,
        "semantic_type": semantic,
        "semantic_confidence": round(float(conf), 2),
        "detection_reason": reason,
        "value_sample": [str(v)[:40] for v in sample.tolist()],
    }


def detect_schema(df: pd.DataFrame, max_columns: int = 60) -> Dict[str, Dict[str, object]]:
    """Detect technical + semantic type for every column (bounded for speed)."""
    out: Dict[str, Dict[str, object]] = {}
    for col in list(df.columns)[:max_columns]:
        out[str(col)] = detect_semantic_type(df[col])
    return out

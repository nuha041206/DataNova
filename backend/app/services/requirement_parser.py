"""
DataNova "Tell DataNova What You Want" — natural-language requirement parser
(Mode B).

Parses free text into a structured, EDITABLE RequirementSpec:

    rows, unique flag, country context, task, target,
    columns[]  {name, technical_type, semantic_type, min, max,
                categories, distribution, unique},
    class_distribution, constraints, domain, clarifications, warnings

This is a deterministic rules engine (regex + keyword matching + schema
mapping) — NOT a trained ML/LLM model, and it does not claim to be one. It is
transparent, testable and every parsed field can be edited in the UI before
generation. Ambiguity produces explicit clarification questions instead of
silent guesses.

Example
-------
Input:
    "Create 2000 unique Indian customer records with name, gender, age,
     address, city, state, Indian phone number, email and salary.
     Age should be between 18 and 65."

Output (abridged):
    rows: 2000, unique: True, country: India
    Name -> string / person_name
    Gender -> categorical / gender
    Age -> integer / age (18-65)
    Address -> string / address
    City -> string / city
    State -> categorical / state
    Phone -> string / indian_phone
    Email -> string / email
    Salary -> float / salary
"""
from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from .types import RequirementSpec
from .semantic_types import SEMANTIC_LABELS, _NAME_HINTS
from .semantic_generators import _STATE_CITIES

_TASK_WORDS_CLASSIFICATION = (
    "classification", "classify", "predict whether", "churn", "fraud", "approval",
    "approved", "default", "diagnosis", "spam", "pass/fail", "risk category",
)
_TASK_WORDS_REGRESSION = (
    "regression", "predict the", "forecast", "estimate the", "price prediction",
    "predict price", "predict value", "score prediction", "amount prediction",
)

# ---------------------------------------------------------------------------
# Column-phrase -> (label, semantic_type)
# ---------------------------------------------------------------------------

_PHRASE_MAP: List[Tuple[Tuple[str, ...], str, str]] = [
    # (phrases, display label, semantic type)
    (("indian phone number", "indian mobile number", "phone number", "phone no",
      "phone", "mobile number", "mobile", "contact number"), "Phone", "indian_phone"),
    (("e-mail", "email", "email address", "mail id"), "Email", "email"),
    (("full name", "customer name", "employee name", "student name", "person name",
      "first name", "last name", "name"), "Name", "person_name"),
    (("gender",), "Gender", "gender"),
    (("address", "home address", "residential address", "street address"), "Address", "address"),
    (("city", "town"), "City", "city"),
    (("state",), "State", "state"),
    (("country",), "Country", "country"),
    (("pin code", "pincode", "postal code", "zip code"), "Pincode", "indian_pincode"),
    (("date of birth", "dob", "birth date"), "Date of Birth", "date_of_birth"),
    (("age",), "Age", "age"),
    (("salary", "monthly salary", "annual salary", "income", "wage"), "Salary", "salary"),
    (("customer id", "customer no", "cust id"), "Customer ID", "customer_id"),
    (("product name", "product"), "Product", "product_name"),
    (("company name", "company", "employer"), "Company", "company_name"),
    (("percentage", "percent", "score %"), "Percentage", "percentage"),
    (("date",), "Date", "date"),
    (("datetime", "timestamp"), "Datetime", "datetime"),
    (("tenure", "experience", "years of experience", "months"), "Tenure", "integer"),
    (("monthly charges", "charges", "fees", "bill amount", "expenses"), "Charges", "currency"),
    (("credit score", "score", "rating"), "Score", "percentage"),
    (("marks", "grade points"), "Marks", "integer"),
]

# filler words removed when building a display label from an unknown phrase
# (also used to decide whether a phrase is only "a known phrase + filler")
_FILLER = {"number", "no", "the", "a", "an", "of", "column", "columns", "field",
           "fields", "feature", "features", "values", "records", "data", "with",
           "and", "rows", "row", "information", "details", "etc", "dataset",
           "datasets", "table"}

# Row-count words, including common misspellings, so "1000 rws" still counts.
_ROW_WORDS = (r"rows?|records?|entries|samples|examples|data\s*points|"
              r"rws|rwos|recods|recrods|recs|entires|exampls")

# A fragment of a column list that is really a dataset-size statement
# ("500 rows of realistic Indian data"), not a requested column.
_ROW_COUNT_FRAGMENT = re.compile(r"\d[\d,.\s]*\s*(?:" + _ROW_WORDS + r")\b", re.IGNORECASE)

# Words that show the user is asking DataNova to produce a dataset. Used only
# when nothing could be extracted from the prompt, to separate a vague-but-real
# request from nonsense.
_DATA_INTENT = (
    "dataset", "data set", "data", "records", "rows", "columns", "column", "table",
    "csv", "sample", "generate", "generated", "generation", "create", "created",
    "make", "making", "build", "want", "need", "give", "gimme", "provide", "prepare",
    "produce", "show", "including", "containing", "consisting", "with", "list of",
)

# Keyboard-smash / placeholder patterns that mark a prompt as meaningless.
_NONSENSE_PATTERNS = ("asdf", "qwer", "wert", "erty", "zxcv", "ghjk", "sdfg", "werty",
                      "blah", "lorem", "ipsum", "abcabc", "xyzzy")

_INVALID_PROMPT_MESSAGE = (
    "I couldn't read that as a dataset description. Please say which columns you want and "
    "(optionally) how many rows — for example: \"Create 500 rows with name, age, mobile "
    "number, email, city and salary for Indian data.\""
)

# technical type implied by each semantic type
_TECH_FOR_SEMANTIC = {
    "person_name": "string", "gender": "categorical", "address": "string",
    "city": "string", "state": "categorical", "country": "categorical",
    "indian_phone": "string", "email": "string", "indian_pincode": "string",
    "date_of_birth": "date", "date": "date", "datetime": "datetime",
    "age": "integer", "salary": "float", "currency": "float",
    "percentage": "float", "customer_id": "string", "generic_id": "string",
    "product_name": "string", "company_name": "string", "category": "categorical",
    "boolean": "boolean", "integer": "integer", "float": "float",
    "generic_text": "string",
}

# names that strongly suggest a categorical column even without "values such as"
_CATEGORY_NAME_HINTS = ("department", "type", "status", "category", "segment",
                        "plan", "grade", "level", "tier", "region", "mode",
                        "method", "employment", "education", "qualification")


def _label_for(phrase: str, semantic: str, explicit_label: Optional[str]) -> str:
    """
    Column label: the canonical DataNova label when the phrase IS a known
    phrase (or a known phrase plus filler words only — "salary columns" ->
    "Salary"), otherwise the user's own wording is kept ("date of joining" ->
    "Date Of Joining", "loyalty_tier" -> "Loyalty Tier") so the generated
    dataset keeps the column names that were actually requested.
    """
    if explicit_label:
        return explicit_label
    words = [w for w in re.split(r"[\s_]+", phrase.strip().lower()) if w]
    word_set = set(words)
    for keys, label, _sem in _PHRASE_MAP:
        for k in keys:
            key_words = k.split()
            if words == key_words:
                return label
            # "salary columns" = key + filler only -> canonical label
            if key_words and set(key_words) <= word_set and word_set - set(key_words) <= _FILLER:
                return label
    return " ".join(w.capitalize() for w in words if w not in _FILLER) or phrase.title()


def _normalize(raw: str) -> str:
    p = re.sub(r"[^a-z0-9 %]", " ", raw.lower())
    return re.sub(r"\s+", " ", p).strip()


def _fuzzy_phrase(p: str) -> Optional[Tuple[str, str]]:
    """Closest known phrase when the input is a misspelling ("mobile numbaer")."""
    if len(p) < 4:
        return None
    best: Optional[Tuple[str, str]] = None
    best_ratio = 0.0
    for keys, _label, sem in _PHRASE_MAP:
        for k in keys:
            if len(k) < 4 or abs(len(k) - len(p)) > max(3, len(k) // 3):
                continue
            ratio = SequenceMatcher(None, p, k).ratio()
            if ratio > best_ratio:
                best_ratio, best = ratio, (k, sem)
    if best is not None and best_ratio >= 0.90:
        return best
    return None


def _name_hint_words(p: str, multiword_only: bool = False) -> Optional[str]:
    """Whole-word column-name hints ("employee id" -> customer_id).

    Whole words only, so "filename" does not become a person name.
    `multiword_only` further restricts the match to phrases like "employee id",
    for use when scanning a wider window of words.
    """
    for keys, semantic in _NAME_HINTS:
        for k in keys:
            key = k.replace("_", " ").strip()
            if not key or (multiword_only and " " not in key):
                continue
            if re.search(r"(?<![a-z0-9])" + re.escape(key) + r"(?![a-z0-9])", p):
                return semantic
    return None


def _match_phrase(raw: str) -> Tuple[str, str]:
    """Map a column phrase to (normalised phrase, semantic type)."""
    p = _normalize(raw)
    if not p:
        return "", "generic_text"

    # 1) exact known phrase (e.g. "indian phone number" -> indian_phone)
    for keys, _label, sem in _PHRASE_MAP:
        for k in keys:
            if p == k:
                return p, sem

    # 2) misspelled known phrase ("mobile numbaer" -> Phone / indian_phone)
    fuzzy = _fuzzy_phrase(p)
    if fuzzy is not None:
        return fuzzy

    # 3) a known phrase inside the phrase ("date of joining" -> date)
    best_key, best_sem = None, None
    for keys, _label, sem in _PHRASE_MAP:
        for k in keys:
            if len(k) >= 4 and k in p and (best_key is None or len(k) > len(best_key)):
                best_key, best_sem = k, sem
    if best_key is not None:
        return p, best_sem

    # 4) category-ish names ("department", "contract type")
    if any(h in p for h in _CATEGORY_NAME_HINTS):
        return p, "category"

    # 5) shared name-hint table from semantic detection (whole words)
    hint = _name_hint_words(p)
    if hint:
        return p, hint

    return p, "generic_text"


def _match_phrase_strict(raw: str, wide: bool = False) -> Tuple[str, str]:
    """Like `_match_phrase`, but never matches a single known word INSIDE a
    longer fragment.

    Used when scanning a bare word list ("name age mobile numbaer") so that
    "age" does not swallow the words around it into one junk column. `wide`
    marks a window of two or more words, which may only match a whole known
    phrase, a misspelling of one, or a category/name hint that spans the start
    of the window ("department type", "employee id").
    """
    p = _normalize(raw)
    if not p:
        return "", "generic_text"
    for keys, _label, sem in _PHRASE_MAP:
        for k in keys:
            if p == k:
                return p, sem
    fuzzy = _fuzzy_phrase(p)
    if fuzzy is not None:
        return fuzzy
    if wide:
        starts = "|".join(sorted((re.escape(h) for h in _CATEGORY_NAME_HINTS), key=len, reverse=True))
        if re.match(r"(?:" + starts + r")\b", p):
            return p, "category"
        hint = _name_hint_words(p, multiword_only=True)
    else:
        if any(h in p for h in _CATEGORY_NAME_HINTS):
            return p, "category"
        hint = _name_hint_words(p)
    if hint:
        return p, hint
    return p, "generic_text"


def _extract_rows(text: str) -> Optional[int]:
    low = text.lower()
    # "2000 unique Indian customer records" / "5,000 rows" / "2k records"
    m = re.search(r"(\d{1,3}(?:,\d{3})+|\d+)\s*(k)?\s*(?:" + _ROW_WORDS + r")", low)
    if not m:
        m = re.search(r"(\d{1,3}(?:,\d{3})+|\d+)\s*(k)?\s*(unique\s+)?(indian\s+|us\s+|american\s+)?(customer|employee|student|person|people|user)s?\s*(records|rows|entries|profiles)", low)
    if not m:
        m2 = re.search(r"(\d+)\s*k\s*(?:rows|records|samples|customers|employees|students)", low)
        if m2:
            return int(m2.group(1)) * 1000
        return None
    n = int(m.group(1).replace(",", ""))
    return n * 1000 if m.group(2) else n


def _extract_unique(text: str) -> bool:
    return bool(re.search(r"\bunique\b|\bdistinct\b|\bno duplicates\b", text.lower()))


def _extract_country(text: str) -> Optional[str]:
    low = text.lower()
    if re.search(r"\bindian\b|\bindia\b|\bdesi\b", low):
        return "India"
    m = re.search(r"\b(us|usa|united states|american)\b", low)
    if m:
        return "United States"
    m = re.search(r"\b(uk|british|united kingdom)\b", low)
    if m:
        return "United Kingdom"
    return None


def _extract_columns(text: str) -> List[Dict[str, Any]]:
    """
    Extract requested columns from list phrases:
        "with name, gender, age, address, city, state, phone number, email and salary"
        "containing department with values such as HR, Finance, IT, Sales"
    """
    low = text.lower()
    found: List[Dict[str, Any]] = []
    seen: set = set()

    def add(phrase: str, explicit_label: Optional[str] = None,
            categories: Optional[List[str]] = None) -> None:
        norm, semantic = _match_phrase(phrase)
        if not norm or norm in _FILLER:
            return
        label = _label_for(norm, semantic, explicit_label)
        key = label.lower()
        if key in seen:
            return
        seen.add(key)
        col: Dict[str, Any] = {
            "name": label,
            "technical_type": _TECH_FOR_SEMANTIC.get(semantic, "string"),
            "semantic_type": semantic,
            "min": None,
            "max": None,
            "categories": categories,
            "distribution": None,
            "unique": False,
            "detected": True,
        }
        found.append(col)

    # Pass 1: "Department with values such as HR, Finance" / "Department: HR, Finance"
    # (runs FIRST so the explicit category list wins over the generic list pass).
    # Runs on the ORIGINAL text so category casing (HR, IT) is preserved.
    captured_values: set = set()

    def _cats_from(vals: str) -> List[str]:
        """Split a value list, stopping when a fragment is itself a known column phrase."""
        cats: List[str] = []
        for frag in re.split(r",| and | / ", vals):
            frag = frag.strip()
            if not frag:
                continue
            _norm, sem = _match_phrase(frag)
            if sem != "generic_text" and len(cats) >= 2:
                break  # "... HR, Finance, IT, Sales, age ..." -> stop before age
            cat = frag if (frag.isupper() or frag[0].isupper()) else frag.title()
            cats.append(cat)
            captured_values.add(cat.lower())
        return cats[:12]

    # Value clauses: "<col> with values such as A, B" / "<col>: A, B". The value
    # region stops at the NEXT clause (lookahead) so one sentence can carry
    # several lists: "...gender with values such as Male and Female, age ... and
    # department with values such as HR and Finance."
    _MARKER = r"(?:with|having|of)?\s*values?\s*(?:such as|like|including|:)"
    _NEXT_CLAUSE = r"(?=\s+[A-Za-z][A-Za-z ]{2,30}?\s+" + _MARKER + r"|\.|$)"
    for m in re.finditer(
            r"([A-Za-z][A-Za-z ]{2,30}?)\s*" + _MARKER +
            r"\s*(" + r"[^.]" + r"*?" + r")" + _NEXT_CLAUSE, text):
        col_phrase, vals = m.group(1).strip(), m.group(2)
        cats = _cats_from(vals)
        if cats and len(cats) >= 2:
            add(col_phrase.lower(), categories=cats)
    for m in re.finditer(r"([A-Z][A-Za-z ]{2,30}):\s*([A-Z][^.]+?)(?:\.|$)", text):
        col_phrase, vals = m.group(1).strip(), m.group(2)
        cats = [v.strip() for v in re.split(r",| and ", vals) if v.strip()][:12]
        if 2 <= len(cats):
            captured_values.update(c.lower() for c in cats)
            add(col_phrase.lower(), explicit_label=col_phrase, categories=cats)

    # Pass 2: "columns/fields/features ... X, Y and Z"
    patterns = [
        r"with (?:columns? |fields? |features? |details? |info(?:rmation)? )?(?:like |including )?([^.]+?)(?:\.|$|;)",
        r"include (?:columns? |fields? |features? )?([^.]+?)(?:\.|$|;)",
        r"containing ([^.]+?)(?:\.|$|;)",
        r"consisting of ([^.]+?)(?:\.|$|;)",
    ]
    for pat in patterns:
        m = re.search(pat, low)
        if not m:
            continue
        chunk = m.group(1)
        for stop in (" the target", " target should", " and the target", " where", " such that"):
            if stop in chunk:
                chunk = chunk.split(stop)[0]
        # Strip numeric-range phrases so "age between 18 and 65" / "age from
        # 18 to 65" yield the column 'age' instead of junk columns '18', '65'.
        chunk = re.sub(r"\b(?:between|from)\s+\d{1,9}\s+(?:and|to|\u2013)\s+\d{1,9}\b", " ", chunk)
        chunk = re.sub(r"\b\d{1,9}\s*(?:-|to|\u2013)\s*\d{1,9}\b", " ", chunk)
        parts = re.split(r",| and | & ", chunk)
        for part in parts:
            part = part.strip(" .")
            part = re.sub(r"^(the|a|an|of|with|for|their)\s+", "", part)
            if not part or not (1 < len(part) < 40) or part in ("etc", "e.g"):
                continue
            # Skip dataset-size statements, not column requests:
            # "3000 rows", "500 rows of realistic Indian data", "with 2000 records"
            if re.fullmatch(r"[\d, .\s]*(?:" + _ROW_WORDS + r")?", part, re.IGNORECASE):
                continue
            if _ROW_COUNT_FRAGMENT.search(part) or re.match(r"(?:" + _ROW_WORDS + r")\b", part, re.IGNORECASE):
                continue
            # value-list fragments already captured in pass 1
            if re.search(r"values?\s*(?:such as|like|including|:)", part):
                part = re.split(r"values?\s*(?:such as|like|including|:)", part)[0].strip()
                if part:
                    add(part)
                continue
            if part.lower() in captured_values:
                continue  # value of an earlier categorical column, not a column
            add(part)

    # Fallback: a bare word list with no "with/include/..." lead-in
    # ("name age mobile numbaer", "give me name, age and email"). Only phrases
    # that ARE recognised column names are added, so gibberish and ordinary
    # prose never create junk columns.
    if not found:
        tokens = re.findall(r"[a-z0-9]+", low)
        i = 0
        while i < len(tokens):
            hit = False
            for width in (3, 2, 1):
                if i + width > len(tokens):
                    continue
                candidate = " ".join(tokens[i:i + width])
                _norm, sem = _match_phrase_strict(candidate, wide=width >= 2)
                if sem != "generic_text" and _norm and _norm not in _FILLER:
                    add(candidate)
                    i += width
                    hit = True
                    break
            if not hit:
                i += 1

    return found


def _apply_ranges(text: str, columns: List[Dict[str, Any]]) -> None:
    """Attach min/max from phrases like 'Age should be between 18 and 65'."""
    low = text.lower()

    def col_by_word(word: str) -> Optional[Dict[str, Any]]:
        for c in columns:
            if word and word in c["name"].lower():
                return c
        return None

    # "<col> between X and Y"
    for m in re.finditer(r"([a-z]+)[a-z ]*?(?:should|must|is)?\s*(?:be\s+)?between (\d{1,9}) and (\d{1,9})", low):
        col = col_by_word(m.group(1))
        if col is not None:
            col["min"], col["max"] = int(m.group(2)), int(m.group(3))

    # "<col> X-Y" / "<col> from X to Y"
    for m in re.finditer(r"\b(age|salary|income|price|amount|marks|score)s?\s*(?:of|from|:)?\s*(\d{1,9})\s*(?:-|to|–)\s*(\d{1,9})", low):
        col = col_by_word(m.group(1))
        if col is not None:
            col["min"], col["max"] = int(m.group(2)), int(m.group(3))

    # bare "between X and Y" attached to the only numeric-ish column
    if not any(c["min"] is not None for c in columns):
        m = re.search(r"between (\d{1,9}) and (\d{1,9})", low)
        if m:
            numeric_cols = [c for c in columns
                            if c["semantic_type"] in ("age", "salary", "currency", "percentage",
                                                      "integer", "float")]
            if len(numeric_cols) == 1:
                numeric_cols[0]["min"], numeric_cols[0]["max"] = int(m.group(1)), int(m.group(2))


def _apply_distributions(text: str, columns: List[Dict[str, Any]]) -> None:
    """Attach requested skews: 'Most customers should be from Maharashtra', '60% from Mumbai'."""
    low = text.lower()
    dist_cols = [c for c in columns if c["semantic_type"] in ("state", "city", "category", "gender", "country")]

    def _as_written(val_lower: str) -> str:
        """Keep the user's own casing ("IT" must not become "It")."""
        m = re.search(re.escape(val_lower.strip()), text, re.IGNORECASE)
        return m.group(0).strip() if m else val_lower.strip().title()

    def _holds(semantic: str, value: str) -> bool:
        """Is this value something the column can actually contain?"""
        if semantic == "gender":
            return value.lower() in ("male", "female", "other", "m", "f")
        if semantic in ("state", "city"):
            low_val = value.lower()
            if semantic == "state":
                return any(low_val == s.lower() for s in _STATE_CITIES)
            return any(low_val == city.lower()
                       for cities in _STATE_CITIES.values() for city in cities)
        return True  # category / country: any requested value is allowed

    def _assign(val: str, pct: int) -> None:
        """Attach the skew to the column that can hold it (not just the first one)."""
        val = re.sub(r"^(?:should\s+be|must\s+be|are|is|be|from|to|in|belong\s+to|of)+\s+",
                     "", (val or "").strip(), flags=re.IGNORECASE).strip()
        if not val:
            return
        value = _as_written(val)
        # 1) a column whose own name mentions the value, 2) a column whose value
        # pool contains it ("Mumbai" is a city, not a state), 3) any category.
        target = next((c for c in dist_cols if value.lower() in c["name"].lower()), None)
        if target is None:
            target = next((c for c in dist_cols if _holds(c["semantic_type"], value)), None)
        if target is not None:
            target["distribution"] = {value: pct}

    # "X% from <place>" / "X% should be <value>"
    for m in re.finditer(r"(\d{1,3})\s*%\s*(?:of (?:the )?\w+ )?(?:should be |are |from |belong to )?([a-z][a-z ]{2,30})", low):
        pct, val = int(m.group(1)), m.group(2).strip()
        if val.lower().rstrip("s") in ("customer", "employee", "student", "record", "people", "user"):
            continue
        _assign(val, pct)

    # "most <rows> should be from <place>"
    m = re.search(r"most (?:of the )?\w+ (?:should|must)?\s*(?:be )?from ([a-z][a-z ]{2,30}?)(?:\.|,|$| and)", low)
    if m:
        _assign(m.group(1), 60)


def _extract_task(text: str) -> str:
    low = text.lower()
    if any(w in low for w in _TASK_WORDS_REGRESSION) and not any(w in low for w in _TASK_WORDS_CLASSIFICATION):
        return "regression"
    if any(w in low for w in _TASK_WORDS_CLASSIFICATION):
        return "classification"
    if "predict" in low:
        return "unknown"
    return "unknown"


def _extract_target(text: str, columns: List[Dict[str, Any]]) -> Optional[str]:
    low = text.lower()
    patterns = [
        r"target (?:should be|is|:)\s*([a-z0-9_ ]+?)(?:\.|,| with|$)",
        r"target column\s*(?:is|:)?\s*([a-z0-9_ ]+?)(?:\.|,| with|$)",
        r"predict (?:whether [a-z ]+ )?([a-z0-9_ ]+?)(?:\.|,| for| using|$)",
        r"to predict ([a-z0-9_ ]+?)(?:\.|,| for| using|$)",
    ]
    for pat in patterns:
        m = re.search(pat, low)
        if m:
            cand = m.group(1).strip()
            cand = re.sub(r"^(the|a|an)\s+", "", cand).split(" for ")[0].strip()
            if cand and len(cand) < 40:
                for c in columns:
                    if cand.split()[0] in c["name"].lower():
                        return c["name"]
                return cand  # preserve the user's own casing
    return None


def _extract_class_distribution(text: str) -> Optional[Dict[str, Any]]:
    low = text.lower()
    m = re.search(r"(\d{1,3})\s*%\s*(?:positive|churn|fraud|default|approved|minority|true|1)", low)
    if m:
        return {"positive_pct": int(m.group(1)), "kind": "positive_share"}
    m = re.search(r"(\d{1,3})\s*%\s*(?:negative|majority|false|0)", low)
    if m:
        return {"positive_pct": 100 - int(m.group(1)), "kind": "positive_share"}
    return None


def _extract_domain(text: str) -> Optional[str]:
    low = text.lower()
    domains = {
        "churn": "customer churn", "loan": "loan approval", "credit": "credit risk",
        "fraud": "fraud detection", "medical": "medical", "patient": "medical",
        "disease": "medical", "student": "education", "exam": "education",
        "employee": "hr analytics", "salary": "hr analytics",
        "customer": "customer data", "insurance": "insurance", "house": "real estate",
        "sales": "sales", "product": "products",
    }
    for key, label in domains.items():
        if key in low:
            return label
    return None


def _build_clarifications(columns: List[Dict[str, Any]], rows: Optional[int],
                          task: str, text: str) -> List[Dict[str, Any]]:
    questions: List[Dict[str, Any]] = []
    if not rows:
        questions.append({
            "field": "rows",
            "question": "How many rows should the dataset contain?",
            "suggestion": 2000,
            "options": [500, 1000, 2000, 5000],
        })
    if not columns:
        questions.append({
            "field": "columns",
            "question": "Which columns should the dataset contain? Add them in the schema editor below (with a semantic type for each).",
            "suggestion": None,
            "options": [],
        })
    if task == "unknown" and any(c["semantic_type"] in ("age", "salary", "currency", "float", "integer") for c in columns):
        questions.append({
            "field": "task",
            "question": "Should this be a classification or a regression dataset? (Skip if you don't need an ML target.)",
            "suggestion": "none",
            "options": ["classification", "regression", "none"],
        })
    if not re.search(r"\bindian\b|\bindia\b", text.lower()) and any(
            c["semantic_type"] in ("indian_phone", "indian_pincode") for c in columns):
        questions.append({
            "field": "country",
            "question": "Should phone numbers / pincodes follow the Indian format?",
            "suggestion": "yes",
            "options": ["yes", "no"],
        })
    return questions


def _looks_like_nonsense(text: str) -> bool:
    """
    Heuristic for meaningless input, only consulted when NOTHING was extracted
    (no columns, rows, country, task, target or domain). Keeps genuinely valid
    but informal prompts working ("I want some data." stays valid).
    """
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    if len(tokens) < 2:
        return True
    counts = Counter(tokens)
    if any(c >= 3 for c in counts.values()):        # "blah blah blah ..."
        return True
    for t in tokens:
        if any(pat in t for pat in _NONSENSE_PATTERNS):   # "asdfghjkl", "qwerty", "lorem"
            return True
        if len(t) >= 6:                             # vowel-poor keyboard smash
            vowels = sum(ch in "aeiou" for ch in t)
            if vowels / len(t) <= 0.15:
                return True
    return False


def _has_data_intent(text: str) -> bool:
    """A request to produce a dataset, or at least a concrete size to build on."""
    low = (text or "").lower()
    if any(w in low for w in _DATA_INTENT):
        return True
    return bool(re.search(r"\b\d{1,7}\b", low)) and len(re.findall(r"[a-z]{3,}", low)) >= 2


def parse_requirements(text: str) -> RequirementSpec:
    """
    Main entry point: free text -> structured, editable RequirementSpec.

    Raises ValueError with a user-facing message when the text is empty or
    cannot be interpreted as a dataset request at all, so gibberish is never
    silently turned into a random dataset.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError(_INVALID_PROMPT_MESSAGE)
    warnings: List[str] = []

    rows = _extract_rows(text)
    unique = _extract_unique(text)
    country = _extract_country(text)
    columns = _extract_columns(text)
    _apply_ranges(text, columns)
    _apply_distributions(text, columns)
    task = _extract_task(text)
    target = _extract_target(text, columns)
    class_distribution = _extract_class_distribution(text)
    domain = _extract_domain(text)

    # Nothing at all could be understood -> say so instead of inventing a
    # dataset. An unfamiliar-but-meaningful column name still parses (it lands
    # in `columns`), so it is never rejected here.
    if (not columns and rows is None and country is None and task == "unknown"
            and target is None and domain is None
            and (_looks_like_nonsense(text) or not _has_data_intent(text))):
        raise ValueError(_INVALID_PROMPT_MESSAGE)

    # Impossible range ("between 65 and 18") -> swap and tell the user.
    for col in columns:
        if col["min"] is not None and col["max"] is not None and col["min"] > col["max"]:
            col["min"], col["max"] = col["max"], col["min"]
            warnings.append(
                f"The range on '{col['name']}' looked reversed (min > max); it was swapped to "
                f"{col['min']}-{col['max']}."
            )

    # India context: Indian semantic types are generated as such
    if country == "India" and not any(c["semantic_type"].startswith("indian_") for c in columns):
        pass  # names/cities already Indian-flavoured by default

    if rows and rows > 200_000:
        warnings.append("Requested rows exceed the 200,000 cap; it will be reduced.")
        rows = 200_000
    if unique:
        warnings.append("Uniqueness is ON: DataNova will remove duplicates and regenerate until the requested count is unique (or explain why it cannot).")

    spec = RequirementSpec(
        rows=rows or 0,
        unique=unique,
        country=country,
        task=task,
        target=target,
        columns=columns,
        constraints=[],
        class_distribution=class_distribution,
        domain=domain,
        raw_text=text,
        warnings=warnings,
    )
    spec.clarifications = _build_clarifications(columns, rows, task, text)
    return spec


def spec_to_dict(spec: RequirementSpec) -> Dict[str, Any]:
    """Serialise the spec for the API response."""
    return {
        "rows": spec.rows,
        "unique": spec.unique,
        "country": spec.country,
        "task": spec.task,
        "target": spec.target,
        "columns": spec.columns,
        "constraints": spec.constraints,
        "class_distribution": spec.class_distribution,
        "domain": spec.domain,
        "clarifications": spec.clarifications,
        "warnings": spec.warnings,
        "raw_text": spec.raw_text,
    }

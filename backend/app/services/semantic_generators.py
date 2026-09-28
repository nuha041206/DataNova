"""
DataNova semantic data generators.

One small class per semantic type, all sharing the same interface:
    generate(n, rng, context) -> list of values

`context` may carry per-column configuration the parser/UI extracted:
    {"min": 18, "max": 65}                       for age / numbers
    {"categories": [...], "weights": [...]}      for gender / category
    {"distribution": {"Maharashtra": 60}}        for state with a requested skew

These are deterministic, rule-based synthetic generators — no ML model is
claimed. Indian variants (names, cities, states, pincodes, phones, addresses)
are supported explicitly so DataNova is useful for Indian datasets. All values
are clearly synthetic: emails use example.com / example.org; addresses are
assembled from common street patterns, NOT real locations.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Value pools (small, clearly synthetic, India-focused)
# ---------------------------------------------------------------------------

_FIRST_NAMES = [
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh", "Krishna",
    "Ishaan", "Shaurya", "Ananya", "Diya", "Aadhya", "Kiara", "Myra", "Anika",
    "Navya", "Sara", "Riya", "Pari", "Rahul", "Rohit", "Amit", "Vikram",
    "Suresh", "Ramesh", "Priya", "Sneha", "Pooja", "Neha", "Kavya", "Divya",
    "Meera", "Anjali", "Kiran", "Manish", "Deepak", "Pankaj", "Nitin", "Ajay",
    "Sunita", "Rekha", "Geeta", "Seema", "Vandana", "Shreya", "Swara", "Tanvi",
]

_LAST_NAMES = [
    "Sharma", "Patel", "Singh", "Kumar", "Gupta", "Reddy", "Rao", "Mehta",
    "Joshi", "Desai", "Iyer", "Nair", "Verma", "Malhotra", "Kapoor", "Shah",
    "Das", "Banerjee", "Mukherjee", "Chatterjee", "Menon", "Pillai", "Bose",
    "Ghosh", "Pandey", "Mishra", "Yadav", "Chauhan", "Rathod", "Naik",
]

_MALE_FIRST = [n for n in _FIRST_NAMES[:20] + ["Karan", "Nikhil", "Sameer", "Arnav", "Dev"]]
_FEMALE_FIRST = [n for n in _FIRST_NAMES[20:40] + ["Ira", "Tara", "Zoya", "Mira", "Anaya"]]

_STREET_NAMES = [
    "MG Road", "Station Road", "Park Street", "Ring Road", "Mall Road",
    "Hill Road", "Temple Street", "Market Road", "Church Road", "Lake View Road",
]

# Generic localities that exist (in name) across most Indian cities, so an
# address stays plausible no matter which city is paired with it.
_LOCALITIES = [
    "Gandhi Nagar", "Nehru Nagar", "Civil Lines", "Rajendra Nagar",
    "Ashok Nagar", "Shastri Nagar", "Sector 12", "Model Town",
    "Vijay Nagar", "Raviwar Peth",
]

_STATE_CITIES: Dict[str, List[str]] = {
    "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Nashik", "Aurangabad", "Thane"],
    "Delhi": ["New Delhi", "Dwarka", "Rohini", "Saket"],
    "Karnataka": ["Bengaluru", "Mysuru", "Mangaluru", "Hubballi"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Tiruchirappalli"],
    "Telangana": ["Hyderabad", "Warangal", "Nizamabad"],
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot"],
    "West Bengal": ["Kolkata", "Howrah", "Durgapur", "Siliguri"],
    "Uttar Pradesh": ["Lucknow", "Kanpur", "Noida", "Varanasi", "Agra"],
    "Rajasthan": ["Jaipur", "Jodhpur", "Udaipur", "Kota"],
    "Kerala": ["Kochi", "Thiruvananthapuram", "Kozhikode", "Thrissur"],
    "Punjab": ["Ludhiana", "Amritsar", "Jalandhar", "Mohali"],
    "Haryana": ["Gurugram", "Faridabad", "Panipat"],
    "Madhya Pradesh": ["Indore", "Bhopal", "Gwalior", "Jabalpur"],
}

# PIN-code first-two-digit zones roughly aligned with states (synthetic, not
# guaranteed accurate to real postal boundaries).
_STATE_PIN_PREFIX: Dict[str, List[str]] = {
    "Maharashtra": ["400", "411", "422", "440"],
    "Delhi": ["110"],
    "Karnataka": ["560", "561", "570"],
    "Tamil Nadu": ["600", "641", "625"],
    "Telangana": ["500", "505"],
    "Gujarat": ["380", "395", "390"],
    "West Bengal": ["700", "711", "712"],
    "Uttar Pradesh": ["226", "208", "201", "221"],
    "Rajasthan": ["302", "342", "313"],
    "Kerala": ["682", "695", "673"],
    "Punjab": ["141", "143", "160"],
    "Haryana": ["122", "121"],
    "Madhya Pradesh": ["452", "462", "458"],
    "Bihar": ["800", "801"],
    "Odisha": ["751", "752"],
}

_MOBILE_PREFIXES = [9, 8, 7, 6]  # valid Indian mobile starting digits


# ---------------------------------------------------------------------------
# Generator interface
# ---------------------------------------------------------------------------

class SemanticGenerator:
    """Base class. Subclasses implement generate()."""
    semantic_type = "generic_text"

    def generate(self, n: int, rng: random.Random, context: Optional[Dict[str, Any]] = None) -> List[Any]:
        raise NotImplementedError


class NameGenerator(SemanticGenerator):
    """Plausible synthetic Indian names (first + surname)."""
    semantic_type = "person_name"

    def generate(self, n, rng, context=None):
        gender = (context or {}).get("gender_series")  # optional coordination
        out: List[Any] = []
        for i in range(n):
            g = gender[i] if gender is not None else None
            if g == "Male":
                first = rng.choice(_MALE_FIRST)
            elif g == "Female":
                first = rng.choice(_FEMALE_FIRST)
            else:
                first = rng.choice(_FIRST_NAMES)
            out.append(f"{first} {rng.choice(_LAST_NAMES)}")
        return out


class GenderGenerator(SemanticGenerator):
    """Gender categories with a configurable distribution."""
    semantic_type = "gender"

    def generate(self, n, rng, context=None):
        cats = (context or {}).get("categories") or ["Male", "Female", "Other"]
        weights = (context or {}).get("weights")
        return rng.choices(cats, weights=weights, k=n)


class PhoneGenerator(SemanticGenerator):
    """Valid-looking Indian 10-digit mobile numbers (starts 6-9)."""
    semantic_type = "indian_phone"

    def generate(self, n, rng, context=None):
        return [f"{rng.choice(_MOBILE_PREFIXES)}{rng.randint(0, 999999999):09d}" for _ in range(n)]


class PincodeGenerator(SemanticGenerator):
    """Six-digit Indian PIN codes, optionally aligned to a state series."""
    semantic_type = "indian_pincode"

    def generate(self, n, rng, context=None):
        state_series = (context or {}).get("state_series")
        out: List[Any] = []
        for i in range(n):
            prefix = None
            if state_series is not None:
                st = state_series[i]
                prefixes = _STATE_PIN_PREFIX.get(st)
                if prefixes:
                    prefix = rng.choice(prefixes)
            if prefix:
                out.append(f"{prefix}{rng.randint(0, 999):03d}")
            else:
                out.append(f"{rng.randint(1, 8)}{rng.randint(0, 99999):05d}")
        return out


class EmailGenerator(SemanticGenerator):
    """Syntactically valid synthetic emails on reserved example domains."""
    semantic_type = "email"

    def generate(self, n, rng, context=None):
        names = (context or {}).get("name_series")
        domains = ["example.com", "example.org", "example.net"]
        out: List[Any] = []
        for i in range(n):
            base = names[i] if names is not None else rng.choice(_FIRST_NAMES)
            slug = base.lower().replace(" ", ".").replace("..", ".")
            slug = "".join(ch for ch in slug if ch.isalpha() or ch == ".")
            out.append(f"{slug}{rng.randint(1, 999)}@{rng.choice(domains)}")
        return out


class AddressGenerator(SemanticGenerator):
    """Synthetic address strings; NOT real addresses."""
    semantic_type = "address"

    def generate(self, n, rng, context=None):
        city_series = (context or {}).get("city_series")
        state_series = (context or {}).get("state_series")
        out: List[Any] = []
        for i in range(n):
            house = rng.randint(1, 999)
            street = rng.choice(_STREET_NAMES)
            locality = rng.choice(_LOCALITIES)
            city = city_series[i] if city_series is not None else rng.choice(list(_STATE_CITIES.items()))[1][0]
            state = state_series[i] if state_series is not None else ""
            tail = f", {state}" if state else ""
            out.append(f"{house} {street}, {locality}, {city}{tail}")
        return out


class CityGenerator(SemanticGenerator):
    """Indian cities, optionally coordinated with the state column."""
    semantic_type = "city"

    def generate(self, n, rng, context=None):
        state_series = (context or {}).get("state_series")
        if state_series is not None:
            states = list(_STATE_CITIES.keys())
            return [
                rng.choice(_STATE_CITIES.get(state_series[i], _STATE_CITIES[states[i % len(states)]]))
                for i in range(n)
            ]
        pool = (context or {}).get("categories")
        if pool:
            return rng.choices(pool, k=n)
        all_cities = [c for cities in _STATE_CITIES.values() for c in cities]
        return [rng.choice(all_cities) for _ in range(n)]


class StateGenerator(SemanticGenerator):
    """Indian states/UTs with optional requested distribution."""
    semantic_type = "state"

    def generate(self, n, rng, context=None):
        dist = (context or {}).get("distribution")  # {"Maharashtra": 60}
        if dist:
            states = list(dist.keys())
            weights = list(dist.values())
            return rng.choices(states, weights=weights, k=n)
        return rng.choices(list(_STATE_CITIES.keys()), k=n)


class CountryGenerator(SemanticGenerator):
    semantic_type = "country"

    def generate(self, n, rng, context=None):
        pool = (context or {}).get("categories") or ["India"]
        if len(pool) == 1:
            return [pool[0]] * n
        return rng.choices(pool, k=n)


class AgeGenerator(SemanticGenerator):
    """Integer ages in a configurable range (default 18-65, slight skew)."""
    semantic_type = "age"

    def generate(self, n, rng, context=None):
        ctx = context or {}
        lo = int(ctx.get("min") if ctx.get("min") is not None else 18)
        hi = int(ctx.get("max") if ctx.get("max") is not None else 65)
        # Triangular-ish distribution centred in the range (more realistic
        # than uniform, still fully within bounds).
        mid = (lo + hi) / 2
        vals = [int(round(rng.triangular(lo, hi, mid))) for _ in range(n)]
        return vals


class SalaryGenerator(SemanticGenerator):
    """Right-skewed salary in INR; optionally correlated with age."""
    semantic_type = "salary"

    def generate(self, n, rng, context=None):
        ctx = context or {}
        age_series = ctx.get("age_series")
        lo = float(ctx.get("min") if ctx.get("min") is not None else 200000)
        hi = float(ctx.get("max") if ctx.get("max") is not None else 2500000)
        out: List[Any] = []
        for i in range(n):
            base = rng.uniform(0.15, 1.0)
            if age_series is not None:
                # older -> moderately higher salary (weak relationship)
                norm = (float(age_series[i]) - 18) / 47.0
                base = min(1.0, base * 0.6 + norm * 0.55)
            val = lo + base * (hi - lo)
            out.append(round(val, -3))  # round to nearest 1000
        return out


class CurrencyGenerator(SemanticGenerator):
    """Generic positive currency amounts."""
    semantic_type = "currency"

    def generate(self, n, rng, context=None):
        ctx = context or {}
        lo = float(ctx.get("min") if ctx.get("min") is not None else 100)
        hi = float(ctx.get("max") if ctx.get("max") is not None else 100000)
        vals = [round(rng.uniform(lo, hi), 2) for _ in range(n)]
        return vals


class PercentageGenerator(SemanticGenerator):
    semantic_type = "percentage"

    def generate(self, n, rng, context=None):
        ctx = context or {}
        as_fraction = bool(ctx.get("fraction", False))
        lo = float(ctx.get("min") if ctx.get("min") is not None else 0)
        hi = float(ctx.get("max") if ctx.get("max") is not None else (100 if not as_fraction else 1))
        return [round(rng.uniform(lo, hi), 3) for _ in range(n)]


class DateOfBirthGenerator(SemanticGenerator):
    """Dates of birth consistent with a requested age range."""
    semantic_type = "date_of_birth"

    def generate(self, n, rng, context=None):
        from datetime import date, timedelta
        ctx = context or {}
        lo = int(ctx.get("min", 18))
        hi = int(ctx.get("max", 65))
        today = date(2026, 1, 1)
        out: List[Any] = []
        for _ in range(n):
            age = rng.triangular(lo, hi, (lo + hi) / 2)
            born = today - timedelta(days=int(age * 365.25 + rng.randint(0, 364)))
            out.append(born.isoformat())
        return out


class DateGenerator(SemanticGenerator):
    """Generic dates in a recent window."""
    semantic_type = "date"

    def generate(self, n, rng, context=None):
        from datetime import date, timedelta
        start = date(2023, 1, 1)
        return [(start + timedelta(days=rng.randint(0, 1095))).isoformat() for _ in range(n)]


class DatetimeGenerator(DateGenerator):
    semantic_type = "datetime"

    def generate(self, n, rng, context=None):
        from datetime import date, timedelta, time
        start = date(2023, 1, 1)
        out = []
        for _ in range(n):
            d = start + timedelta(days=rng.randint(0, 1095))
            t = time(rng.randint(0, 23), rng.choice((0, 15, 30, 45)))
            out.append(d.isoformat() + " " + t.isoformat())
        return out


class IdGenerator(SemanticGenerator):
    """Prefixed unique IDs: CUST000001, ORD000014, ..."""
    semantic_type = "customer_id"

    def generate(self, n, rng, context=None):
        prefix = str((context or {}).get("prefix", "ID"))
        width = max(4, len(str(n)) + 1)
        return [f"{prefix}{i + 1:0{width}d}" for i in range(n)]


class CompanyNameGenerator(SemanticGenerator):
    """Synthetic company names (pattern-assembled, not real firms)."""
    semantic_type = "company_name"

    def generate(self, n, rng, context=None):
        a = ["Sunrise", "BluePeak", "NovaTech", "GreenLeaf", "Metro", "Apex",
             "Trident", "Orbit", "Vertex", "Silverline", "Prime", "Zenith"]
        b = ["Solutions", "Industries", "Enterprises", "Technologies", "Logistics",
             "Foods", "Textiles", "Motors", "Pharma", "Infotech", "Retail"]
        suffix = ["Pvt Ltd", "LLP", "Ltd", "Corp"]
        return [
            f"{rng.choice(a)} {rng.choice(b)} {rng.choice(suffix)}" for _ in range(n)
        ]


class ProductNameGenerator(SemanticGenerator):
    semantic_type = "product_name"

    def generate(self, n, rng, context=None):
        pool = (context or {}).get("categories")
        if pool:
            return rng.choices(pool, k=n)
        a = ["Aurora", "Nimbus", "Zephyr", "Titan", "Fusion", "Pulse", "Echo", "Nova"]
        b = ["Pro", "Lite", "Max", "Ultra", "Mini", "Plus", "XL"]
        return [f"{rng.choice(a)} {rng.choice(b)}" for _ in range(n)]


class CategoryGenerator(SemanticGenerator):
    """Categorical values from the requested pool (+ optional distribution)."""
    semantic_type = "category"

    def generate(self, n, rng, context=None):
        cats = (context or {}).get("categories") or ["Category A", "Category B", "Category C"]
        dist = (context or {}).get("distribution")
        if dist:
            cats = list(dist.keys())
            return rng.choices(cats, weights=list(dist.values()), k=n)
        return rng.choices(cats, k=n)


class BooleanGenerator(SemanticGenerator):
    semantic_type = "boolean"

    def generate(self, n, rng, context=None):
        true_pct = float((context or {}).get("true_pct", 50)) / 100
        return [rng.random() < true_pct for _ in range(n)]


class IntegerGenerator(SemanticGenerator):
    semantic_type = "integer"

    def generate(self, n, rng, context=None):
        ctx = context or {}
        lo = int(ctx.get("min") if ctx.get("min") is not None else 0)
        hi = int(ctx.get("max") if ctx.get("max") is not None else 100)
        return [rng.randint(lo, hi) for _ in range(n)]


class FloatGenerator(SemanticGenerator):
    semantic_type = "float"

    def generate(self, n, rng, context=None):
        ctx = context or {}
        lo = float(ctx.get("min") if ctx.get("min") is not None else 0)
        hi = float(ctx.get("max") if ctx.get("max") is not None else 100)
        return [round(rng.uniform(lo, hi), 2) for _ in range(n)]


class GenericTextGenerator(SemanticGenerator):
    """
    Controlled synthetic text: short word-pair phrases.

    Honest limitation: DataNova does not generate intelligent free-form
    natural language; this produces plausible filler text.
    """
    semantic_type = "generic_text"

    def generate(self, n, rng, context=None):
        a = ["Quarterly", "Customer", "Regional", "Standard", "Preliminary",
             "Operational", "Annual", "Internal", "Priority", "Special"]
        b = ["review", "report", "summary", "record", "entry", "note",
             "request", "update", "account", "transaction"]
        return [f"{rng.choice(a)} {rng.choice(b)} {rng.randint(1, 99)}" for _ in range(n)]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_GENERATORS: Dict[str, type] = {
    "person_name": NameGenerator,
    "gender": GenderGenerator,
    "indian_phone": PhoneGenerator,
    "email": EmailGenerator,
    "address": AddressGenerator,
    "city": CityGenerator,
    "state": StateGenerator,
    "country": CountryGenerator,
    "indian_pincode": PincodeGenerator,
    "date_of_birth": DateOfBirthGenerator,
    "date": DateGenerator,
    "datetime": DatetimeGenerator,
    "age": AgeGenerator,
    "salary": SalaryGenerator,
    "currency": CurrencyGenerator,
    "percentage": PercentageGenerator,
    "customer_id": IdGenerator,
    "generic_id": IdGenerator,
    "company_name": CompanyNameGenerator,
    "product_name": ProductNameGenerator,
    "category": CategoryGenerator,
    "boolean": BooleanGenerator,
    "integer": IntegerGenerator,
    "float": FloatGenerator,
    "generic_text": GenericTextGenerator,
}


def generator_for(semantic_type: str) -> SemanticGenerator:
    """Return a generator instance for a semantic type (never None)."""
    cls = _GENERATORS.get(semantic_type, GenericTextGenerator)
    return cls()

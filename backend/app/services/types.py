"""
Shared service-layer types for DataNova.

DataNova deliberately uses plain Python dicts as the exchange format between
services and JSON responses in the API. This keeps the code simple and easy to
read for this project. These dataclasses document the *shape* of the
main payloads so the frontend team (and the examiner) can follow the flow:

    upload -> analysis -> detection -> generation -> evaluation -> report
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ColumnProfile:
    """Per-column summary produced by the dataset analyzer."""

    name: str
    dtype: str                      # pandas dtype as string, e.g. "int64"
    inferred_type: str              # semantic type: numerical / categorical / datetime / boolean / identifier
    missing: int
    missing_pct: float
    unique: int
    unique_pct: float
    # Numeric summaries (None for non-numeric columns)
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    q25: Optional[float] = None
    median: Optional[float] = None
    q75: Optional[float] = None
    # Categorical summaries (None for non-categorical columns)
    top_values: Optional[List[Dict[str, Any]]] = None  # [{"value": v, "count": n}, ...]
    # Health flags
    is_constant: bool = False
    is_high_cardinality: bool = False
    looks_like_id: bool = False


@dataclass
class TargetSuggestion:
    """Result of automatic target detection."""

    column: Optional[str]           # None when nothing plausible was found
    task: str                       # "classification" or "regression"
    confidence: str                 # "high" | "medium" | "low"
    score: float                    # 0-1 internal heuristic score
    reason: str                     # human-readable explanation


@dataclass
class RequirementSpec:
    """Structured requirement produced by the Mode B NL parser."""

    rows: int
    task: str                       # classification | regression | unknown
    target: Optional[str]
    features: List[Dict[str, Any]] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    class_distribution: Optional[Dict[str, Any]] = None
    domain: Optional[str] = None
    clarifications: List[Dict[str, Any]] = field(default_factory=list)
    raw_text: str = ""
    warnings: List[str] = field(default_factory=list)
    # Semantic schema (v2): full column descriptors with technical + semantic types
    columns: List[Dict[str, Any]] = field(default_factory=list)
    unique: bool = False            # requested unique records
    country: Optional[str] = None   # geographic context, e.g. India


@dataclass
class GenerationResult:
    """Payload returned by the generator service."""

    dataframe_preview: List[Dict[str, Any]]
    row_count: int
    columns: List[str]
    approach: str                   # which synthesizer was used
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

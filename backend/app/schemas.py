"""
Pydantic request/response schemas for the DataNova API.

Kept in one module so the API contract is easy to review.
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DetectRequest(BaseModel):
    """Re-run target detection with an optional manual override."""
    target: Optional[str] = None


class GenerateRequest(BaseModel):
    """SDV generation parameters (Mode A)."""
    n_rows: int = Field(gt=0, description="How many synthetic rows to generate")
    target: Optional[str] = None
    task: Optional[str] = None
    exclude_columns: Optional[List[str]] = None
    synthesizer: Optional[str] = None  # override: GaussianCopula | CTGAN | TVAE
    advanced: Optional[Dict[str, Any]] = None  # e.g. {"epochs": 200}


class EvaluateRequest(BaseModel):
    """Which pair of frames to evaluate."""
    upload_id: Optional[str] = None
    project_id: Optional[int] = None
    target: Optional[str] = None


class MLRequest(BaseModel):
    """Model recommendation request."""
    upload_id: Optional[str] = None
    task: str
    target: str


class MLRunRequest(BaseModel):
    """Run the ML utility experiments."""
    upload_id: Optional[str] = None
    project_id: Optional[int] = None
    target: str
    task: str
    model_key: Optional[str] = None


class ExplainRequest(BaseModel):
    """Inputs for the explainability engine (already-computed results)."""
    evaluation: Optional[Dict[str, Any]] = None
    ml_utility: Optional[Dict[str, Any]] = None
    generation_approach: Optional[str] = None
    generation_reason: Optional[str] = None
    warnings: Optional[List[str]] = None


class ParseRequest(BaseModel):
    """Free-text requirements for Mode B."""
    text: str


class ColumnSpec(BaseModel):
    """One column in a requirement spec: technical + semantic type + rules."""
    name: str
    technical_type: str = "string"   # string|integer|float|boolean|date|datetime|categorical
    semantic_type: str = "generic_text"
    min: Optional[float] = None
    max: Optional[float] = None
    categories: Optional[List[str]] = None
    distribution: Optional[Dict[str, Any]] = None
    unique: bool = False
    detected: bool = False           # True when inferred automatically


class FeatureSpec(BaseModel):
    """Legacy feature descriptor (kept for backward compatibility)."""
    name: str
    type: str = "float"  # integer | float | category | boolean


class RequirementSpecModel(BaseModel):
    """Editable structured requirement (Mode B)."""
    rows: int = 0
    task: str = "unknown"
    target: Optional[str] = None
    features: List[FeatureSpec] = []
    constraints: List[str] = []
    class_distribution: Optional[Dict[str, Any]] = None
    domain: Optional[str] = None
    raw_text: str = ""
    warnings: List[str] = []
    # Semantic schema (v2)
    columns: List[ColumnSpec] = []
    unique: bool = False
    country: Optional[str] = None


class RequirementGenerateRequest(BaseModel):
    """Generate from a confirmed/edited spec."""
    spec: RequirementSpecModel
    seed: Optional[int] = 42


class SaveProjectRequest(BaseModel):
    """Persist the workflow as a project."""
    name: str
    mode: str
    # Session id whose generated synthetic CSV should be persisted to disk.
    # Optional because requirement-mode projects have no session frames.
    upload_id: Optional[str] = None
    dataset_filename: Optional[str] = None
    rows_real: Optional[int] = None
    columns: Optional[int] = None
    target: Optional[str] = None
    task: Optional[str] = None
    target_overridden: bool = False
    synthesizer: Optional[str] = None
    generation_reason: Optional[str] = None
    rows_requested: Optional[int] = None
    rows_synthetic: Optional[int] = None
    fit_seconds: Optional[float] = None
    sample_seconds: Optional[float] = None
    analysis: Optional[Dict[str, Any]] = None
    detection: Optional[Dict[str, Any]] = None
    evaluation: Optional[Dict[str, Any]] = None
    ml_utility: Optional[Dict[str, Any]] = None
    explanation: Optional[Dict[str, Any]] = None
    requirement_spec: Optional[Dict[str, Any]] = None
    warnings: Optional[List[str]] = None


class RenameRequest(BaseModel):
    """Rename a saved project (downloads and future reports follow the name)."""
    name: str


class ReportRequest(BaseModel):
    """Reserved for future report options (e.g. sections to include)."""
    include_sections: Optional[List[str]] = None

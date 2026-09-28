"""
DataNova persistence layer (SQLite via SQLAlchemy).

Stores one row per project with the full result payloads (analysis, evaluation,
ml_utility, explanation, requirement spec) as JSON text columns. For a
single-user local application this is the right simplicity trade-off: one table,
easy to inspect with any SQLite browser, no migration tooling.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import String, Integer, Float, Text, create_engine, insert, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from . import config


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""


class Project(Base):
    """A single DataNova project (one full workflow run)."""

    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200))
    mode: Mapped[str] = mapped_column(String(20))                  # upload | requirements
    created_at: Mapped[str] = mapped_column(String(30), default=lambda: datetime.now().isoformat(timespec="seconds"))

    # Dataset info
    dataset_filename: Mapped[Optional[str]] = mapped_column(String(300), nullable=True)
    rows_real: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    columns: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    target: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    task: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    target_overridden: Mapped[Optional[bool]] = mapped_column(nullable=True)

    # Generation info
    synthesizer: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    generation_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    rows_requested: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    rows_synthetic: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    fit_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sample_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Requirement-based mode payload
    requirement_spec: Mapped[Optional[str]] = mapped_column(Text, nullable=True)   # JSON

    # Result payloads (JSON strings)
    analysis: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    detection: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    evaluation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    ml_utility: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    explanation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    warnings: Mapped[Optional[str]] = mapped_column(Text, nullable=True)           # JSON list

    # Report info
    report_path: Mapped[Optional[str]] = mapped_column(String(400), nullable=True)
    report_generated_at: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)

    # Synth data preview (first N rows as JSON) so history can show a taste
    synth_preview: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Messy dataset (JSON summary + file name). Files live in data/synthetic.
    messy_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    messy_file: Mapped[Optional[str]] = mapped_column(String(300), nullable=True)

    # Session id of the persisted upload (data/uploads/{upload_ref}.csv) so
    # evaluation / ML utility can reload the real dataset for saved projects.
    upload_ref: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    def to_dict(self, include_payloads: bool = True) -> Dict[str, Any]:
        """Serialise the row; payloads parsed back into dicts when requested."""
        d: Dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "mode": self.mode,
            "created_at": self.created_at,
            "dataset_filename": self.dataset_filename,
            "rows_real": self.rows_real,
            "columns": self.columns,
            "target": self.target,
            "task": self.task,
            "target_overridden": self.target_overridden,
            "synthesizer": self.synthesizer,
            "generation_reason": self.generation_reason,
            "rows_requested": self.rows_requested,
            "rows_synthetic": self.rows_synthetic,
            "fit_seconds": self.fit_seconds,
            "sample_seconds": self.sample_seconds,
            "report_generated_at": self.report_generated_at,
            "report_path": self.report_path,
            "has_report": bool(self.report_path),
            "messy_file": self.messy_file,
            "upload_ref": self.upload_ref,
        }
        if include_payloads:
            for field in ("analysis", "detection", "evaluation", "ml_utility",
                          "explanation", "requirement_spec", "synth_preview"):
                raw = getattr(self, field)
                d[field] = json.loads(raw) if raw else None
            d["warnings"] = json.loads(self.warnings) if self.warnings else []
            d["messy"] = json.loads(self.messy_summary) if self.messy_summary else None
        d["has_messy"] = bool(self.messy_file)
        return d


_engine = None
_SessionLocal = None


def get_engine():
    """Lazily create the SQLite engine (creates folders on first use)."""
    global _engine, _SessionLocal
    if _engine is None:
        config.ensure_dirs()
        _engine = create_engine(f"sqlite:///{config.DB_PATH}", echo=False, future=True)
        Base.metadata.create_all(_engine)
        _ensure_schema(_engine)
        _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def _ensure_schema(engine) -> None:
    """Add columns introduced after the first release (idempotent, tiny)."""
    from sqlalchemy import text
    with engine.connect() as conn:
        cols = {r[1] for r in conn.execute(text("PRAGMA table_info(projects)"))}
        for name, ddl in (("messy_summary", "TEXT"), ("messy_file", "VARCHAR(300)"),
                          ("upload_ref", "VARCHAR(100)")):
            if name not in cols:
                conn.execute(text(f"ALTER TABLE projects ADD COLUMN {name} {ddl}"))
        conn.commit()


def get_session():
    """Return a new session (caller closes it)."""
    get_engine()
    return _SessionLocal()


# ---------------------------------------------------------------------------
# Repository helpers (kept tiny on purpose)
# ---------------------------------------------------------------------------

def save_project(fields: Dict[str, Any]) -> int:
    """Insert a new project row and return its id."""
    with get_session() as session:
        row = Project(**fields)
        session.add(row)
        session.commit()
        return row.id


def update_project(project_id: int, fields: Dict[str, Any]) -> None:
    """Update selected columns of a project row (JSON values must be serialised)."""
    with get_session() as session:
        row = session.get(Project, project_id)
        if row is None:
            raise ValueError(f"Project {project_id} not found")
        for k, v in fields.items():
            setattr(row, k, v)
        session.commit()


def list_projects() -> List[Dict[str, Any]]:
    """All projects, newest first, without heavy payloads."""
    with get_session() as session:
        rows = session.execute(
            select(Project).order_by(Project.id.desc())
        ).scalars().all()
        return [r.to_dict(include_payloads=False) for r in rows]


def get_project(project_id: int) -> Optional[Dict[str, Any]]:
    """One project with full payloads, or None."""
    with get_session() as session:
        row = session.get(Project, project_id)
        return row.to_dict(include_payloads=True) if row else None


def delete_project(project_id: int) -> bool:
    """Remove a project row; returns True when something was deleted."""
    with get_session() as session:
        row = session.get(Project, project_id)
        if row is None:
            return False
        session.delete(row)
        session.commit()
        return True

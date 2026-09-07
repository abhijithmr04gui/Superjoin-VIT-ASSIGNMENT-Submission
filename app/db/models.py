"""
Persistent schema for the Fact Knowledge Layer.

Design notes:
- Facts use a generic subject/predicate/object-ish shape (entity, predicate,
  value) plus optional structured fields (unit, currency, dates, scope,
  qualifiers) rather than one column per fact "type" (revenue, director...).
  New fact types need zero schema changes.
- Every Fact row carries its own evidence (source text + page + offsets) so
  a fact is never stored without a way to trace it back to the PDF.
- Relationships are stored between two Fact ids, not baked into Fact rows,
  so a fact can participate in many relationships as the knowledge base grows.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


def _uid() -> str:
    return uuid.uuid4().hex[:16]


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uid)
    filename: Mapped[str] = mapped_column(String)
    file_hash: Mapped[str] = mapped_column(String, index=True)
    stored_path: Mapped[str] = mapped_column(String)
    num_pages: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String, default="uploaded")  # uploaded/processing/completed/failed
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    doc_metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    chunks: Mapped[list["Chunk"]] = relationship(back_populates="document", cascade="all, delete-orphan")
    facts: Mapped[list["Fact"]] = relationship(back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uid)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), index=True)
    page_number: Mapped[int] = mapped_column(Integer)
    chunk_index: Mapped[int] = mapped_column(Integer)
    section_header: Mapped[str | None] = mapped_column(String, nullable=True)
    text: Mapped[str] = mapped_column(Text)
    char_start: Mapped[int] = mapped_column(Integer, default=0)
    char_end: Mapped[int] = mapped_column(Integer, default=0)
    embedding: Mapped[list] = mapped_column(JSON, default=list)  # dense vector, see vector_store
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    document: Mapped["Document"] = relationship(back_populates="chunks")


class Fact(Base):
    __tablename__ = "facts"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uid)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), index=True)
    chunk_id: Mapped[str | None] = mapped_column(ForeignKey("chunks.id"), nullable=True)

    # Core generic triple-ish shape.
    entity: Mapped[str] = mapped_column(String, index=True)
    predicate: Mapped[str] = mapped_column(String, index=True)
    object_text: Mapped[str] = mapped_column(Text)  # the raw stated value, human-readable

    fact_type: Mapped[str] = mapped_column(String, default="semantic")  # numeric/semantic/entity/temporal/scoped
    value_type: Mapped[str] = mapped_column(String, default="text")  # number/text/date/boolean

    normalized_value: Mapped[dict] = mapped_column(JSON, default=dict)  # {"number": 10000000, ...}
    unit: Mapped[str | None] = mapped_column(String, nullable=True)
    currency: Mapped[str | None] = mapped_column(String, nullable=True)

    date: Mapped[str | None] = mapped_column(String, nullable=True)  # ISO string, point-in-time facts
    start_date: Mapped[str | None] = mapped_column(String, nullable=True)
    end_date: Mapped[str | None] = mapped_column(String, nullable=True)
    reporting_period: Mapped[str | None] = mapped_column(String, nullable=True)  # e.g. "FY2024"

    scope: Mapped[str | None] = mapped_column(String, nullable=True)  # e.g. "North America"
    location: Mapped[str | None] = mapped_column(String, nullable=True)
    qualifiers: Mapped[list] = mapped_column(JSON, default=list)  # ["excluding acquisition revenue"]

    # Evidence - never optional for a valid fact.
    source_text: Mapped[str] = mapped_column(Text)
    page_number: Mapped[int] = mapped_column(Integer)
    char_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    char_end: Mapped[int | None] = mapped_column(Integer, nullable=True)

    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    confidence_level: Mapped[str] = mapped_column(String, default="MEDIUM")  # HIGH/MEDIUM/LOW

    extraction_method: Mapped[str] = mapped_column(String, default="llm")
    is_valid: Mapped[bool] = mapped_column(Boolean, default=True)
    validation_notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    document: Mapped["Document"] = relationship(back_populates="facts")


class FactRelationship(Base):
    __tablename__ = "fact_relationships"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uid)
    fact_a_id: Mapped[str] = mapped_column(ForeignKey("facts.id"), index=True)
    fact_b_id: Mapped[str] = mapped_column(ForeignKey("facts.id"), index=True)

    relationship_type: Mapped[str] = mapped_column(String)
    # CORROBORATED / CONTRADICTED / LIKELY_CONTRADICTION /
    # CONTEXTUALLY_RECONCILED / UNCERTAIN / UNRELATED

    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    confidence_level: Mapped[str] = mapped_column(String, default="MEDIUM")

    explanation: Mapped[str] = mapped_column(Text)
    contextual_dimensions: Mapped[dict] = mapped_column(JSON, default=dict)
    # e.g. {"time": "compatible", "scope": "differs", "resolved_by": "reporting_period"}

    reasoning_method: Mapped[str] = mapped_column(String, default="llm")  # deterministic/llm/hybrid
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class ProcessingRun(Base):
    __tablename__ = "processing_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uid)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), index=True)
    status: Mapped[str] = mapped_column(String, default="running")  # running/completed/failed
    current_stage: Mapped[str | None] = mapped_column(String, nullable=True)
    facts_extracted: Mapped[int] = mapped_column(Integer, default=0)
    relationships_created: Mapped[int] = mapped_column(Integer, default=0)
    error_count: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class ExtractionFailure(Base):
    """
    Explicit record of an extraction or reasoning failure. This is a
    first-class table, not a log line, so failures are queryable and
    visible through the API/UI (assignment section 18 / case 4).
    """

    __tablename__ = "extraction_failures"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uid)
    run_id: Mapped[str | None] = mapped_column(ForeignKey("processing_runs.id"), nullable=True)
    document_id: Mapped[str | None] = mapped_column(ForeignKey("documents.id"), nullable=True)
    chunk_id: Mapped[str | None] = mapped_column(String, nullable=True)
    fact_id: Mapped[str | None] = mapped_column(String, nullable=True)

    stage: Mapped[str] = mapped_column(String)  # extraction/validation/normalization/comparison/...
    error_type: Mapped[str] = mapped_column(String)
    error_message: Mapped[str] = mapped_column(Text)
    severity: Mapped[str] = mapped_column(String, default="medium")  # low/medium/high

    recovery_attempted: Mapped[bool] = mapped_column(Boolean, default=False)
    recovery_successful: Mapped[bool] = mapped_column(Boolean, default=False)
    final_status: Mapped[str] = mapped_column(String, default="unresolved")  # resolved/unresolved/discarded
    suggested_improvement: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

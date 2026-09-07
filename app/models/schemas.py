from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class DocumentOut(BaseModel):
    id: str
    filename: str
    num_pages: int
    status: str
    error_message: str | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class EvidenceOut(BaseModel):
    document_id: str
    filename: str
    page_number: int
    chunk_id: str | None
    source_text: str
    char_start: int | None
    char_end: int | None


class FactOut(BaseModel):
    id: str
    document_id: str
    entity: str
    predicate: str
    object_text: str
    fact_type: str
    value_type: str
    normalized_value: dict[str, Any]
    unit: str | None
    currency: str | None
    date: str | None
    start_date: str | None
    end_date: str | None
    reporting_period: str | None
    scope: str | None
    location: str | None
    qualifiers: list[str]
    confidence: float
    confidence_level: str
    extraction_method: str
    is_valid: bool
    validation_notes: str | None
    evidence: EvidenceOut

    class Config:
        from_attributes = True


class RelationshipOut(BaseModel):
    id: str
    fact_a: FactOut
    fact_b: FactOut
    relationship_type: str
    confidence: float
    confidence_level: str
    explanation: str
    contextual_dimensions: dict[str, Any]
    reasoning_method: str

    class Config:
        from_attributes = True


class FailureOut(BaseModel):
    id: str
    document_id: str | None
    stage: str
    error_type: str
    error_message: str
    severity: str
    recovery_attempted: bool
    recovery_successful: bool
    final_status: str
    suggested_improvement: str | None
    created_at: datetime

    class Config:
        from_attributes = True


class ProcessingRunOut(BaseModel):
    id: str
    document_id: str
    status: str
    current_stage: str | None
    facts_extracted: int
    relationships_created: int
    error_count: int
    started_at: datetime
    completed_at: datetime | None

    class Config:
        from_attributes = True


class UploadResponse(BaseModel):
    document: DocumentOut
    run: ProcessingRunOut


class SearchResult(BaseModel):
    fact: FactOut
    score: float

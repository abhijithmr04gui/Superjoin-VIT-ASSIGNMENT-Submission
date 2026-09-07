from __future__ import annotations

from app.db.models import Fact, FactRelationship
from app.models.schemas import EvidenceOut, FactOut, RelationshipOut


def fact_to_schema(fact: Fact) -> FactOut:
    evidence = EvidenceOut(
        document_id=fact.document_id,
        filename=fact.document.filename if fact.document else "",
        page_number=fact.page_number,
        chunk_id=fact.chunk_id,
        source_text=fact.source_text,
        char_start=fact.char_start,
        char_end=fact.char_end,
    )
    return FactOut(
        id=fact.id,
        document_id=fact.document_id,
        entity=fact.entity,
        predicate=fact.predicate,
        object_text=fact.object_text,
        fact_type=fact.fact_type,
        value_type=fact.value_type,
        normalized_value=fact.normalized_value or {},
        unit=fact.unit,
        currency=fact.currency,
        date=fact.date,
        start_date=fact.start_date,
        end_date=fact.end_date,
        reporting_period=fact.reporting_period,
        scope=fact.scope,
        location=fact.location,
        qualifiers=fact.qualifiers or [],
        confidence=fact.confidence,
        confidence_level=fact.confidence_level,
        extraction_method=fact.extraction_method,
        is_valid=fact.is_valid,
        validation_notes=fact.validation_notes,
        evidence=evidence,
    )


def relationship_to_schema(rel: FactRelationship, fact_a: Fact, fact_b: Fact) -> RelationshipOut:
    return RelationshipOut(
        id=rel.id,
        fact_a=fact_to_schema(fact_a),
        fact_b=fact_to_schema(fact_b),
        relationship_type=rel.relationship_type,
        confidence=rel.confidence,
        confidence_level=rel.confidence_level,
        explanation=rel.explanation,
        contextual_dimensions=rel.contextual_dimensions or {},
        reasoning_method=rel.reasoning_method,
    )

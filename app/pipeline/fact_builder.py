from __future__ import annotations

from dataclasses import dataclass

from app.pipeline.extraction import RawExtractedFact
from app.pipeline.normalization import normalize_date, normalize_number
from app.pipeline.validation import ValidationResult, validate_fact


def _confidence_level(score: float) -> str:
    if score >= 0.75:
        return "HIGH"
    if score >= 0.45:
        return "MEDIUM"
    return "LOW"


@dataclass
class BuiltFact:
    entity: str
    predicate: str
    object_text: str
    fact_type: str
    value_type: str
    normalized_value: dict
    unit: str | None
    currency: str | None
    date: str | None
    start_date: str | None
    end_date: str | None
    reporting_period: str | None
    scope: str | None
    location: str | None
    qualifiers: list[str]
    source_text: str
    page_number: int
    char_start: int | None
    char_end: int | None
    confidence: float
    confidence_level: str
    is_valid: bool
    validation_notes: str


def build_fact(raw: RawExtractedFact, chunk_text: str, page_number: int, num_pages: int) -> BuiltFact:
    validation: ValidationResult = validate_fact(raw, chunk_text, page_number, num_pages)

    normalized_value: dict = {}
    unit = raw.unit
    currency = None
    date_norm = raw.date
    reporting_period = raw.reporting_period

    if raw.value_type == "number" or raw.fact_type == "numeric":
        num_result = normalize_number(raw.object_text)
        normalized_value = {"number": num_result.value, "parse_ok": num_result.parse_ok, "notes": num_result.notes}
        currency = num_result.currency
        unit = unit or num_result.unit

    if raw.date:
        d = normalize_date(raw.date)
        date_norm = d.iso_date
        normalized_value.setdefault("date_precision", d.precision)
    if raw.reporting_period and not reporting_period:
        reporting_period = raw.reporting_period
    if raw.reporting_period:
        d = normalize_date(raw.reporting_period)
        if d.parse_ok and "period_start" not in normalized_value:
            normalized_value["period_start"] = d.iso_date

    # LLM confidence is a starting point; deterministic validation can only
    # lower it, never inflate it (assignment section 16/17: don't overtrust
    # the LLM's self-reported confidence).
    confidence = raw.confidence
    if not validation.is_valid:
        confidence = min(confidence, 0.2)
    elif "fuzzy" in validation.notes:
        confidence = min(confidence, 0.6)

    return BuiltFact(
        entity=raw.entity,
        predicate=raw.predicate,
        object_text=raw.object_text,
        fact_type=raw.fact_type,
        value_type=raw.value_type,
        normalized_value=normalized_value,
        unit=unit,
        currency=currency,
        date=date_norm,
        start_date=raw.start_date,
        end_date=raw.end_date,
        reporting_period=reporting_period,
        scope=raw.scope,
        location=raw.location,
        qualifiers=raw.qualifiers,
        source_text=raw.source_text,
        page_number=page_number,
        char_start=validation.char_start,
        char_end=validation.char_end,
        confidence=confidence,
        confidence_level=_confidence_level(confidence),
        is_valid=validation.is_valid,
        validation_notes=validation.notes,
    )

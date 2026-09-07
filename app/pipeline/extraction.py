"""
LLM-based structured fact extraction (assignment section 9).

This module ONLY calls the LLM and returns raw parsed dicts + any
extraction failure encountered. Validation (section 10) and
normalization (section 11) are deliberately separate stages.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.core.logging import get_logger
from app.pipeline.llm_client import LLMNotConfiguredError, call_json
from prompts import fact_extraction

logger = get_logger("extraction")


@dataclass
class RawExtractedFact:
    entity: str
    predicate: str
    object_text: str
    fact_type: str = "semantic"
    value_type: str = "text"
    unit: str | None = None
    date: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    reporting_period: str | None = None
    scope: str | None = None
    location: str | None = None
    qualifiers: list[str] = field(default_factory=list)
    source_text: str = ""
    confidence: float = 0.5


@dataclass
class ExtractionOutcome:
    facts: list[RawExtractedFact]
    ok: bool
    error: str | None = None
    raw_response: str = ""


_REQUIRED_KEYS = {"entity", "predicate", "object_text", "source_text"}


def _coerce(item: dict) -> RawExtractedFact | None:
    if not isinstance(item, dict):
        return None
    if not _REQUIRED_KEYS.issubset(item.keys()):
        return None
    try:
        return RawExtractedFact(
            entity=str(item.get("entity", "")).strip(),
            predicate=str(item.get("predicate", "")).strip().lower().replace(" ", "_"),
            object_text=str(item.get("object_text", "")).strip(),
            fact_type=item.get("fact_type") or "semantic",
            value_type=item.get("value_type") or "text",
            unit=item.get("unit"),
            date=item.get("date"),
            start_date=item.get("start_date"),
            end_date=item.get("end_date"),
            reporting_period=item.get("reporting_period"),
            scope=item.get("scope"),
            location=item.get("location"),
            qualifiers=item.get("qualifiers") or [],
            source_text=str(item.get("source_text", "")).strip(),
            confidence=float(item.get("confidence", 0.5) or 0.5),
        )
    except (TypeError, ValueError) as exc:
        logger.warning("Failed to coerce extracted fact item: %s (%s)", item, exc)
        return None


def extract_facts_from_chunk(filename: str, page_number: int, chunk_text: str) -> ExtractionOutcome:
    if not chunk_text.strip():
        return ExtractionOutcome(facts=[], ok=True)

    user_prompt = fact_extraction.USER_PROMPT_TEMPLATE.format(
        filename=filename, page_number=page_number, chunk_text=chunk_text
    )

    try:
        result = call_json(fact_extraction.SYSTEM_PROMPT, user_prompt, max_tokens=2500)
    except LLMNotConfiguredError as exc:
        return ExtractionOutcome(facts=[], ok=False, error=str(exc))

    if not result.ok:
        return ExtractionOutcome(facts=[], ok=False, error=result.error, raw_response=result.raw_text)

    items = result.data if isinstance(result.data, list) else []
    facts = [f for f in (_coerce(item) for item in items) if f is not None]

    dropped = len(items) - len(facts)
    if dropped > 0:
        logger.warning("Dropped %d malformed fact(s) from LLM extraction output", dropped)

    return ExtractionOutcome(facts=facts, ok=True, raw_response=result.raw_text)

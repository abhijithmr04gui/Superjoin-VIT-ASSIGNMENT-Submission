"""
Deterministic validation (assignment section 10). No LLM calls here.

The single most important check: does the claimed source_text actually
appear in the chunk it was supposedly extracted from? This is what
prevents a hallucinated fact from silently being stored as if it were
grounded in the PDF.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from app.pipeline.extraction import RawExtractedFact

_MIN_ENTITY_LEN = 1
_FUZZY_MATCH_THRESHOLD = 0.75


@dataclass
class ValidationResult:
    is_valid: bool
    notes: str
    char_start: int | None
    char_end: int | None


def _normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _find_span(chunk_text: str, source_text: str) -> tuple[int, int] | None:
    """Exact (case/whitespace-insensitive) substring search first."""
    norm_chunk = _normalize_for_match(chunk_text)
    norm_source = _normalize_for_match(source_text)
    if not norm_source:
        return None
    idx = norm_chunk.find(norm_source)
    if idx == -1:
        return None
    # Best-effort mapping back to original offsets: since normalization only
    # collapses whitespace/case, lengths are close enough for evidence display.
    return idx, idx + len(norm_source)


def _fuzzy_supported(chunk_text: str, source_text: str) -> float:
    """
    Containment ratio: how much of source_text's content is matched
    *somewhere* within chunk_text, summed across matching blocks. This is
    deliberately not a symmetric similarity ratio (like plain
    SequenceMatcher.ratio()) - a short claimed quote sitting inside a much
    longer chunk should still score high, since the question is "is this
    supported by the chunk", not "are these two strings alike overall".
    """
    norm_chunk = _normalize_for_match(chunk_text)
    norm_source = _normalize_for_match(source_text)
    if not norm_source:
        return 0.0
    matcher = difflib.SequenceMatcher(None, norm_chunk, norm_source)
    matched_chars = sum(block.size for block in matcher.get_matching_blocks())
    return matched_chars / len(norm_source)


def validate_fact(fact: RawExtractedFact, chunk_text: str, page_number: int, num_pages: int) -> ValidationResult:
    notes: list[str] = []

    if not fact.entity or len(fact.entity) < _MIN_ENTITY_LEN:
        return ValidationResult(is_valid=False, notes="missing/empty entity", char_start=None, char_end=None)

    if not fact.predicate:
        return ValidationResult(is_valid=False, notes="missing predicate", char_start=None, char_end=None)

    if not fact.object_text:
        return ValidationResult(is_valid=False, notes="missing object_text/value", char_start=None, char_end=None)

    if not fact.source_text:
        return ValidationResult(is_valid=False, notes="missing evidence (source_text)", char_start=None, char_end=None)

    if page_number < 1 or page_number > num_pages:
        return ValidationResult(
            is_valid=False, notes=f"invalid page number {page_number} (doc has {num_pages} pages)",
            char_start=None, char_end=None,
        )

    span = _find_span(chunk_text, fact.source_text)
    if span is not None:
        return ValidationResult(is_valid=True, notes="exact evidence match", char_start=span[0], char_end=span[1])

    similarity = _fuzzy_supported(chunk_text, fact.source_text)
    if similarity >= _FUZZY_MATCH_THRESHOLD:
        notes.append(f"evidence not an exact substring; fuzzy similarity={similarity:.2f}")
        return ValidationResult(is_valid=True, notes="; ".join(notes), char_start=None, char_end=None)

    return ValidationResult(
        is_valid=False,
        notes=f"evidence text not found in source chunk (fuzzy similarity={similarity:.2f}) - possible hallucination",
        char_start=None,
        char_end=None,
    )

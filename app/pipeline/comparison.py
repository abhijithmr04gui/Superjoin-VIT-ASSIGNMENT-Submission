"""
Fact comparison / relationship classification (assignment sections 14-16).

Deterministic shortcut first (section 30: "if a relationship can be
resolved deterministically, do not spend an LLM call on it"). Only facts
that pass the deterministic pre-check straight to CORROBORATED skip the
LLM; every other case - including anything that looks like it might be a
contradiction - goes through LLM reasoning, because that judgment call is
exactly what section 16 says not to hand to ad hoc code.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.core.logging import get_logger
from app.db.models import Fact
from app.pipeline.llm_client import LLMNotConfiguredError, call_json
from app.pipeline.normalization import normalize_entity_name
from prompts import fact_comparison

logger = get_logger("comparison")

_VALID_TYPES = {
    "CORROBORATED",
    "CONTRADICTED",
    "LIKELY_CONTRADICTION",
    "CONTEXTUALLY_RECONCILED",
    "UNCERTAIN",
    "UNRELATED",
}


@dataclass
class ComparisonResult:
    relationship_type: str
    confidence: float
    explanation: str
    contextual_dimensions: dict
    reasoning_method: str  # deterministic | llm
    ok: bool = True
    error: str | None = None
    rate_limited: bool = False


def _values_agree(a: Fact, b: Fact, tolerance: float = 0.01) -> bool | None:
    va = (a.normalized_value or {}).get("number")
    vb = (b.normalized_value or {}).get("number")
    if va is None or vb is None:
        return None
    if va == 0 and vb == 0:
        return True
    denom = max(abs(va), abs(vb), 1e-9)
    return abs(va - vb) / denom <= tolerance


def deterministic_precheck(fact_a: Fact, fact_b: Fact) -> ComparisonResult | None:
    """
    Only ever returns a confident CORROBORATED verdict, and only when
    every relevant dimension is an exact, unambiguous match. Everything
    else is left to the LLM stage - this function never returns
    CONTRADICTED, because "the values differ" is not sufficient grounds
    for that classification without contextual reasoning.
    """
    same_entity = normalize_entity_name(fact_a.entity) == normalize_entity_name(fact_b.entity)
    same_predicate = fact_a.predicate == fact_b.predicate
    same_period = (fact_a.reporting_period or fact_a.date) == (fact_b.reporting_period or fact_b.date)
    same_scope = (fact_a.scope or None) == (fact_b.scope or None)
    same_qualifiers = sorted(fact_a.qualifiers or []) == sorted(fact_b.qualifiers or [])

    if not (same_entity and same_predicate and same_period and same_scope and same_qualifiers):
        return None

    values_agree = _values_agree(fact_a, fact_b)
    if values_agree is True:
        return ComparisonResult(
            relationship_type="CORROBORATED",
            confidence=0.9,
            explanation=(
                "Same entity, predicate, reporting period/date, scope, and qualifiers, "
                "with matching normalized numeric values - classified deterministically "
                "without an LLM call."
            ),
            contextual_dimensions={
                "entity": "same", "predicate": "same", "time": "compatible",
                "scope": "compatible", "unit": "compatible", "resolved_by": None,
            },
            reasoning_method="deterministic",
        )
    return None


def _fmt(value) -> str:
    return "null" if value in (None, "") else str(value)


def llm_compare(fact_a: Fact, fact_b: Fact) -> ComparisonResult:
    user_prompt = fact_comparison.USER_PROMPT_TEMPLATE.format(
        doc_a_filename=fact_a.document.filename if fact_a.document else fact_a.document_id,
        page_a=fact_a.page_number,
        entity_a=fact_a.entity, predicate_a=fact_a.predicate, object_a=fact_a.object_text,
        normalized_a=_fmt(fact_a.normalized_value), unit_a=_fmt(fact_a.unit), currency_a=_fmt(fact_a.currency),
        date_a=_fmt(fact_a.date), period_a=_fmt(fact_a.reporting_period), scope_a=_fmt(fact_a.scope),
        qualifiers_a=_fmt(fact_a.qualifiers), source_a=fact_a.source_text,
        doc_b_filename=fact_b.document.filename if fact_b.document else fact_b.document_id,
        page_b=fact_b.page_number,
        entity_b=fact_b.entity, predicate_b=fact_b.predicate, object_b=fact_b.object_text,
        normalized_b=_fmt(fact_b.normalized_value), unit_b=_fmt(fact_b.unit), currency_b=_fmt(fact_b.currency),
        date_b=_fmt(fact_b.date), period_b=_fmt(fact_b.reporting_period), scope_b=_fmt(fact_b.scope),
        qualifiers_b=_fmt(fact_b.qualifiers), source_b=fact_b.source_text,
    )

    try:
        result = call_json(fact_comparison.SYSTEM_PROMPT, user_prompt, max_tokens=800)
    except LLMNotConfiguredError as exc:
        return ComparisonResult(
            relationship_type="UNCERTAIN", confidence=0.0, explanation="",
            contextual_dimensions={}, reasoning_method="llm", ok=False, error=str(exc),
        )

    if not result.ok or not isinstance(result.data, dict):
        return ComparisonResult(
            relationship_type="UNCERTAIN", confidence=0.0, explanation="LLM comparison failed",
            contextual_dimensions={}, reasoning_method="llm", ok=False, error=result.error,
            rate_limited=result.rate_limited,
        )

    rel_type = str(result.data.get("relationship_type", "UNCERTAIN")).upper()
    if rel_type not in _VALID_TYPES:
        rel_type = "UNCERTAIN"

    try:
        confidence = float(result.data.get("confidence", 0.3))
    except (TypeError, ValueError):
        confidence = 0.3

    return ComparisonResult(
        relationship_type=rel_type,
        confidence=confidence,
        explanation=str(result.data.get("explanation", "")),
        contextual_dimensions=result.data.get("contextual_dimensions", {}) or {},
        reasoning_method="llm",
        ok=True,
    )


def compare_facts(fact_a: Fact, fact_b: Fact) -> ComparisonResult:
    shortcut = deterministic_precheck(fact_a, fact_b)
    if shortcut is not None:
        return shortcut
    return llm_compare(fact_a, fact_b)

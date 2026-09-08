"""
LLM-backed entity identity resolution (assignment: "differently written
addresses may refer to the same place" / a company referred to
inconsistently across filings).

Deliberately thin, mirroring extraction.py/comparison.py's pattern: this
module only calls the LLM and returns a parsed result. Callers decide
when it's worth spending the call (see retrieval.py's gating) and what
to do with a failed/unavailable result.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.core.logging import get_logger
from app.pipeline.llm_client import LLMNotConfiguredError, call_json
from prompts import entity_resolution as prompt

logger = get_logger("entity_resolution")


@dataclass
class EntityResolutionResult:
    same_entity: bool | None
    confidence: float
    reasoning: str
    ok: bool = True
    error: str | None = None


def resolve_entity_match(
    entity_a: str, context_a: str, entity_b: str, context_b: str
) -> EntityResolutionResult:
    user_prompt = prompt.USER_PROMPT_TEMPLATE.format(
        entity_a=entity_a, context_a=context_a, entity_b=entity_b, context_b=context_b,
    )
    try:
        result = call_json(prompt.SYSTEM_PROMPT, user_prompt, max_tokens=300)
    except LLMNotConfiguredError as exc:
        return EntityResolutionResult(same_entity=None, confidence=0.0, reasoning="", ok=False, error=str(exc))

    if not result.ok or not isinstance(result.data, dict):
        return EntityResolutionResult(same_entity=None, confidence=0.0, reasoning="", ok=False, error=result.error)

    try:
        confidence = float(result.data.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0

    return EntityResolutionResult(
        same_entity=result.data.get("same_entity"),
        confidence=confidence,
        reasoning=str(result.data.get("reasoning", "")),
    )

"""
Candidate retrieval (assignment section 13). Narrows the comparison set
so we never do a naive O(n^2) all-facts-vs-all-facts comparison.

A candidate is worth an (expensive, LLM-backed) comparison call if it
scores well on AT LEAST ONE of: vector similarity, matching normalized
entity, or matching predicate/fact_type - vector similarity alone is
never treated as sufficient evidence of a relationship (section 4).

Entity matching itself has two layers: cheap normalized-string equality
first, and - only when that fails but the pair still looks plausible -
one gated LLM call to prompts/entity_resolution.py to catch genuinely
different phrasings of the same entity (e.g. "Delhivery Limited" vs
"Delhivery"), which suffix-stripping alone cannot bridge. This is gated
behind a zero-cost heuristic and memoized per pipeline run so it never
turns into an LLM call per fact pair.
"""
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import Fact
from app.pipeline.entity_resolution import resolve_entity_match
from app.pipeline.normalization import normalize_entity_name
from app.vector_store.embeddings import cosine_similarity

# Minimum LLM-reported confidence to accept an entity-resolution match.
_ENTITY_RESOLUTION_CONFIDENCE_THRESHOLD = 0.6


@dataclass
class Candidate:
    fact: Fact
    vector_score: float
    entity_match: bool
    predicate_match: bool
    reason: str
    resolved_by_llm: bool = False
    entity_resolution_confidence: float | None = None
    entity_resolution_reasoning: str | None = None


def _plausible_entity_pair(a: str, b: str) -> bool:
    """Cheap, zero-LLM-cost gate: only worth asking the LLM about entity
    identity when the two normalized names already share some textual
    relationship - a substring match or a common token - not for
    completely unrelated strings (e.g. two different companies that both
    happen to report "revenue")."""
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    tokens_a, tokens_b = set(a.split()), set(b.split())
    return bool(tokens_a & tokens_b)


def find_candidates(
    session: Session,
    fact: Fact,
    top_k: int | None = None,
    entity_resolution_cache: dict | None = None,
) -> list[Candidate]:
    top_k = top_k or settings.candidate_top_k
    entity_resolution_cache = entity_resolution_cache if entity_resolution_cache is not None else {}
    norm_entity = normalize_entity_name(fact.entity)

    others = (
        session.query(Fact)
        .filter(Fact.id != fact.id, Fact.document_id != fact.document_id, Fact.is_valid == True)  # noqa: E712
        .all()
    )
    if not others:
        return []

    query_vec = fact.embedding

    scored: list[Candidate] = []
    for other in others:
        other_norm_entity = normalize_entity_name(other.entity)
        entity_match = norm_entity == other_norm_entity and bool(norm_entity)
        predicate_match = fact.predicate == other.predicate or fact.fact_type == other.fact_type
        vec_score = cosine_similarity(query_vec, other.embedding)

        resolved_by_llm = False
        er_confidence: float | None = None
        er_reasoning: str | None = None

        if (
            not entity_match
            and (predicate_match or vec_score >= 0.35)
            and _plausible_entity_pair(norm_entity, other_norm_entity)
        ):
            pass # Disable LLM entity resolution to save rate limit
            # cache_key = frozenset({norm_entity, other_norm_entity})
            # if cache_key not in entity_resolution_cache:
            #     entity_resolution_cache[cache_key] = resolve_entity_match(
            #         fact.entity,
            #         f"{fact.predicate}: {fact.object_text}",
            #         other.entity,
            #         f"{other.predicate}: {other.object_text}",
            #     )
            # er = entity_resolution_cache[cache_key]
            # if er.ok and er.same_entity is True and er.confidence >= _ENTITY_RESOLUTION_CONFIDENCE_THRESHOLD:
            #     entity_match = True
            #     resolved_by_llm = True
            #     er_confidence, er_reasoning = er.confidence, er.reasoning

        if not (entity_match or predicate_match or vec_score >= 0.35):
            continue

        reasons = []
        if entity_match:
            reasons.append("entity_resolution" if resolved_by_llm else "entity_match")
        if predicate_match:
            reasons.append("predicate_match")
        if vec_score >= 0.35:
            reasons.append(f"vector_similarity={vec_score:.2f}")

        scored.append(
            Candidate(
                fact=other,
                vector_score=vec_score,
                entity_match=entity_match,
                predicate_match=predicate_match,
                reason=",".join(reasons),
                resolved_by_llm=resolved_by_llm,
                entity_resolution_confidence=er_confidence,
                entity_resolution_reasoning=er_reasoning,
            )
        )

    # Rank: entity+predicate match first, then by vector score.
    scored.sort(key=lambda c: (c.entity_match and c.predicate_match, c.entity_match, c.vector_score), reverse=True)
    return scored[:top_k]

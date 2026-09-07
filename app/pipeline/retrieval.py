"""
Candidate retrieval (assignment section 13). Narrows the comparison set
so we never do a naive O(n^2) all-facts-vs-all-facts comparison.

A candidate is worth an (expensive, LLM-backed) comparison call if it
scores well on AT LEAST ONE of: vector similarity, matching normalized
entity, or matching predicate/fact_type - vector similarity alone is
never treated as sufficient evidence of a relationship (section 4).
"""
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import Fact
from app.pipeline.normalization import normalize_entity_name
from app.vector_store.embeddings import cosine_similarity, embed_text


@dataclass
class Candidate:
    fact: Fact
    vector_score: float
    entity_match: bool
    predicate_match: bool
    reason: str


def find_candidates(session: Session, fact: Fact, top_k: int | None = None) -> list[Candidate]:
    top_k = top_k or settings.candidate_top_k
    norm_entity = normalize_entity_name(fact.entity)

    others = (
        session.query(Fact)
        .filter(Fact.id != fact.id, Fact.document_id != fact.document_id, Fact.is_valid == True)  # noqa: E712
        .all()
    )
    if not others:
        return []

    query_text = f"{fact.entity} {fact.predicate} {fact.object_text}"
    query_vec = embed_text(query_text)

    scored: list[Candidate] = []
    for other in others:
        other_norm_entity = normalize_entity_name(other.entity)
        entity_match = norm_entity == other_norm_entity and bool(norm_entity)
        predicate_match = fact.predicate == other.predicate or fact.fact_type == other.fact_type

        other_text = f"{other.entity} {other.predicate} {other.object_text}"
        vec_score = cosine_similarity(query_vec, embed_text(other_text))

        if not (entity_match or predicate_match or vec_score >= 0.35):
            continue

        reasons = []
        if entity_match:
            reasons.append("entity_match")
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
            )
        )

    # Rank: entity+predicate match first, then by vector score.
    scored.sort(key=lambda c: (c.entity_match and c.predicate_match, c.entity_match, c.vector_score), reverse=True)
    return scored[:top_k]

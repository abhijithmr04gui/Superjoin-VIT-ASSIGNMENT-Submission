"""
The "vector database" for this prototype is the embedding column already
present on Chunk/Fact rows in SQLite, queried in-process with numpy cosine
similarity. This satisfies the assignment's own preference ("prefer a
simple locally runnable option unless another choice has a strong
engineering justification" - section 4) and keeps the whole system a
single dependency-light process.

Swapping this module for Chroma/FAISS/pgvector later only requires
changing the function below - nothing in the pipeline depends on the
storage mechanism directly.

Fact embeddings are cached at fact-build time (see fact_builder.py) and
reused here rather than recomputed per search request - see
embeddings.fact_embedding_text for the canonical embedding text.
"""
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.db.models import Fact
from app.vector_store.embeddings import cosine_similarity, embed_text


@dataclass
class ScoredFact:
    fact: Fact
    score: float


def find_similar_facts(
    session: Session,
    query_text: str,
    top_k: int = 8,
    exclude_fact_id: str | None = None,
    exclude_document_id: str | None = None,
) -> list[ScoredFact]:
    query_vec = embed_text(query_text)
    q = session.query(Fact).filter(Fact.is_valid == True)  # noqa: E712
    if exclude_document_id:
        q = q.filter(Fact.document_id != exclude_document_id)
    if exclude_fact_id:
        q = q.filter(Fact.id != exclude_fact_id)
    candidates = q.all()

    scored = [
        ScoredFact(fact=f, score=cosine_similarity(query_vec, f.embedding))
        for f in candidates
        if f.embedding
    ]
    scored.sort(key=lambda s: s.score, reverse=True)
    return scored[:top_k]

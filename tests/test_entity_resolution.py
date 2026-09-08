"""
Tests for LLM-backed entity resolution wiring in retrieval.py. The LLM
call site (app.pipeline.entity_resolution.resolve_entity_match, imported
into retrieval.py) is mocked - no real API calls, consistent with the
rest of the suite.

Covers:
  - a plausible pair with differing entity strings gets resolved and
    flagged (the "Delhivery" vs "Delhivery Logistics Private Limited"
    case: normalize_entity_name alone does not unify these, but they
    share a token so the cheap plausibility gate lets it through);
  - an implausible pair (no token/substring overlap) never triggers the
    LLM call at all;
  - the per-run cache means a repeated entity pair is only resolved once.
"""
from __future__ import annotations

import app.pipeline.retrieval as retrieval_module
from app.db.database import get_session
from app.db.models import Document, Fact
from app.pipeline.entity_resolution import EntityResolutionResult
from app.pipeline.retrieval import find_candidates


def _make_document(session, filename: str) -> str:
    doc = Document(filename=filename, file_hash=f"hash-{filename}", stored_path=f"/tmp/{filename}")
    session.add(doc)
    session.flush()
    return doc.id


def _make_fact(session, document_id: str, entity: str, predicate: str = "revenue", **kwargs) -> Fact:
    fact = Fact(
        document_id=document_id,
        entity=entity,
        predicate=predicate,
        object_text=kwargs.pop("object_text", "$10 million"),
        source_text=kwargs.pop("source_text", "revenue was $10 million"),
        page_number=kwargs.pop("page_number", 1),
        **kwargs,
    )
    session.add(fact)
    session.flush()
    return fact


def test_plausible_pair_resolved_and_flagged(monkeypatch):
    calls = []

    def fake_resolve(entity_a, context_a, entity_b, context_b):
        calls.append((entity_a, entity_b))
        return EntityResolutionResult(same_entity=True, confidence=0.9, reasoning="same company, different legal name")

    monkeypatch.setattr(retrieval_module, "resolve_entity_match", fake_resolve)

    with get_session() as session:
        doc1 = _make_document(session, "delhivery_a.pdf")
        doc2 = _make_document(session, "delhivery_b.pdf")
        fact = _make_fact(session, doc1, "Delhivery")
        other = _make_fact(session, doc2, "Delhivery Logistics Private Limited")

        candidates = find_candidates(session, fact)

    assert len(calls) == 1
    matches = [c for c in candidates if c.fact.id == other.id]
    assert len(matches) == 1
    assert matches[0].resolved_by_llm is True
    assert matches[0].entity_match is True
    assert matches[0].entity_resolution_confidence == 0.9


def test_implausible_pair_never_calls_llm(monkeypatch):
    def fail_if_called(entity_a, context_a, entity_b, context_b):
        raise AssertionError("resolve_entity_match should not be called for an implausible pair")

    monkeypatch.setattr(retrieval_module, "resolve_entity_match", fail_if_called)

    with get_session() as session:
        doc1 = _make_document(session, "unrelated_a.pdf")
        doc2 = _make_document(session, "unrelated_b.pdf")
        fact = _make_fact(session, doc1, "Northwind Traders", predicate="employee_count", object_text="500")
        _make_fact(session, doc2, "Contoso Holdings", predicate="director_name", object_text="Jane Doe")

        # Should not raise - no shared tokens/substring, and no predicate
        # or vector-similarity signal either, so entity resolution is
        # never even attempted for this pair.
        candidates = find_candidates(session, fact)

    assert candidates == []


def test_entity_resolution_cache_avoids_duplicate_calls():
    calls = []

    def fake_resolve(entity_a, context_a, entity_b, context_b):
        calls.append((entity_a, entity_b))
        return EntityResolutionResult(same_entity=True, confidence=0.9, reasoning="same company")

    import app.pipeline.retrieval as mod

    original = mod.resolve_entity_match
    mod.resolve_entity_match = fake_resolve
    try:
        with get_session() as session:
            doc1 = _make_document(session, "cache_a.pdf")
            doc2 = _make_document(session, "cache_b.pdf")
            fact1 = _make_fact(session, doc1, "Delhivery", object_text="$10 million")
            fact3 = _make_fact(session, doc1, "Delhivery", object_text="$12 million")
            _make_fact(session, doc2, "Delhivery Logistics Private Limited", object_text="$10 million")

            cache: dict = {}
            find_candidates(session, fact1, entity_resolution_cache=cache)
            find_candidates(session, fact3, entity_resolution_cache=cache)
    finally:
        mod.resolve_entity_match = original

    assert len(calls) == 1

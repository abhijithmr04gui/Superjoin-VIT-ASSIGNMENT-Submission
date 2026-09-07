"""
End-to-end pipeline test with the LLM mocked out (no network/API key
needed). This exercises ingest -> chunk -> extract -> validate ->
normalize -> retrieve -> compare -> persist for real, and specifically
demonstrates the four required cases from assignment section 25:

  1. CORROBORATION   (docs A & B: same FY2024 revenue, different wording)
  2. CONTRADICTION   (docs A & C: same FY2024 revenue claim, different values)
  3. RECONCILIATION  (docs C & D: same value claim, different fiscal years)
  4. FAILURE         (doc E: LLM "hallucinates" evidence not in the source
                       chunk; the validation stage catches and records it)

The mock stands in for the LLM at the exact two call sites
(extraction.call_json, comparison.call_json) - everything else in the
pipeline (chunking, normalization, validation, candidate retrieval,
persistence, the LangGraph wiring itself) runs for real.
"""
from __future__ import annotations

import json

import pytest

import app.pipeline.comparison as comparison_module
import app.pipeline.extraction as extraction_module
from app.db.database import get_session
from app.db.models import ExtractionFailure, Fact, FactRelationship
from app.pipeline.llm_client import LLMJsonResult
from tests.conftest import make_pdf
from tests.helpers import process_pdf

DOC_A_TEXT = "Annual revenue for Acme Corp in FY2024 was $10 million."
DOC_B_TEXT = "Acme Corp's revenue reached USD 10 million for FY2024."
DOC_C_TEXT = "Acme Corp reported FY2024 revenue of $15 million."
DOC_D_TEXT = "For FY2025, Acme Corp's revenue was $15 million."
DOC_E_TEXT = "The company's headquarters relocated to a new building in 2023."


def _extraction_fact(entity, predicate, object_text, period, source_text):
    return {
        "entity": entity,
        "predicate": predicate,
        "object_text": object_text,
        "fact_type": "numeric",
        "value_type": "number",
        "unit": None,
        "date": None,
        "start_date": None,
        "end_date": None,
        "reporting_period": period,
        "scope": None,
        "location": None,
        "qualifiers": [],
        "source_text": source_text,
        "confidence": 0.9,
    }


def fake_call_json(system_prompt, user_prompt, max_tokens=2000, temperature=0.0):
    # --- Extraction stage ---
    if "fact-extraction engine" in system_prompt:
        if DOC_A_TEXT in user_prompt:
            data = [_extraction_fact("Acme Corp", "revenue", "$10 million", "FY2024", DOC_A_TEXT)]
        elif DOC_B_TEXT in user_prompt:
            data = [_extraction_fact("Acme Corp", "revenue", "USD 10 million", "FY2024", DOC_B_TEXT)]
        elif DOC_C_TEXT in user_prompt:
            data = [_extraction_fact("Acme Corp", "revenue", "$15 million", "FY2024", DOC_C_TEXT)]
        elif DOC_D_TEXT in user_prompt:
            data = [_extraction_fact("Acme Corp", "revenue", "$15 million", "FY2025", DOC_D_TEXT)]
        elif DOC_E_TEXT in user_prompt:
            # Deliberately hallucinated: this "fact" and its source_text were
            # never in the actual chunk. Simulates a real extraction failure.
            data = [
                _extraction_fact(
                    "Acme Corp", "revenue", "$99 million", "FY2099",
                    "revenue for FY2099 was an incredible $99 million",
                )
            ]
        else:
            data = []
        return LLMJsonResult(ok=True, data=data, raw_text=json.dumps(data))

    # --- Comparison stage ---
    if "relationship between two facts" in system_prompt:
        if "FY2024" in user_prompt and "FY2025" in user_prompt:
            data = {
                "relationship_type": "CONTEXTUALLY_RECONCILED",
                "confidence": 0.85,
                "explanation": (
                    "Same entity and metric, but the two facts refer to different "
                    "fiscal years (FY2024 vs FY2025), which explains the differing "
                    "revenue figures."
                ),
                "contextual_dimensions": {
                    "entity": "same", "predicate": "same", "time": "incompatible",
                    "scope": "compatible", "unit": "compatible", "resolved_by": "reporting_period",
                },
            }
        else:
            data = {
                "relationship_type": "CONTRADICTED",
                "confidence": 0.8,
                "explanation": (
                    "Same entity, metric, and reporting period (FY2024), but the "
                    "stated revenue figures conflict ($10 million vs $15 million) "
                    "with no contextual difference to explain the gap."
                ),
                "contextual_dimensions": {
                    "entity": "same", "predicate": "same", "time": "compatible",
                    "scope": "compatible", "unit": "compatible", "resolved_by": None,
                },
            }
        return LLMJsonResult(ok=True, data=data, raw_text=json.dumps(data))

    raise AssertionError(f"Unexpected system prompt in test mock: {system_prompt[:80]}")


@pytest.fixture(autouse=True)
def _mock_llm(monkeypatch):
    monkeypatch.setattr(extraction_module, "call_json", fake_call_json)
    monkeypatch.setattr(comparison_module, "call_json", fake_call_json)


@pytest.fixture(scope="module")
def processed_docs():
    paths = {
        "A": make_pdf([DOC_A_TEXT]),
        "B": make_pdf([DOC_B_TEXT]),
        "C": make_pdf([DOC_C_TEXT]),
        "D": make_pdf([DOC_D_TEXT]),
        "E": make_pdf([DOC_E_TEXT]),
    }
    # This fixture runs with _mock_llm NOT yet applied (different fixture
    # scope), so process documents inside the test functions instead.
    return paths


def _facts_by_period_and_value(session, period, value):
    return (
        session.query(Fact)
        .filter(Fact.reporting_period == period)
        .filter(Fact.is_valid == True)  # noqa: E712
        .all()
    )


def test_four_required_demonstration_cases(monkeypatch):
    monkeypatch.setattr(extraction_module, "call_json", fake_call_json)
    monkeypatch.setattr(comparison_module, "call_json", fake_call_json)

    doc_a_id, _ = process_pdf(make_pdf([DOC_A_TEXT]), "doc_a.pdf")
    doc_b_id, _ = process_pdf(make_pdf([DOC_B_TEXT]), "doc_b.pdf")
    doc_c_id, _ = process_pdf(make_pdf([DOC_C_TEXT]), "doc_c.pdf")
    doc_d_id, _ = process_pdf(make_pdf([DOC_D_TEXT]), "doc_d.pdf")
    doc_e_id, _ = process_pdf(make_pdf([DOC_E_TEXT]), "doc_e.pdf")

    with get_session() as session:
        fact_a = session.query(Fact).filter(Fact.document_id == doc_a_id).one()
        fact_b = session.query(Fact).filter(Fact.document_id == doc_b_id).one()
        fact_c = session.query(Fact).filter(Fact.document_id == doc_c_id).one()
        fact_d = session.query(Fact).filter(Fact.document_id == doc_d_id).one()

        def relationship_between(f1, f2):
            return (
                session.query(FactRelationship)
                .filter(
                    ((FactRelationship.fact_a_id == f1.id) & (FactRelationship.fact_b_id == f2.id))
                    | ((FactRelationship.fact_a_id == f2.id) & (FactRelationship.fact_b_id == f1.id))
                )
                .first()
            )

        # CASE 1: CORROBORATION (A vs B) - resolved deterministically since
        # every dimension matches exactly (see comparison.deterministic_precheck).
        rel_ab = relationship_between(fact_a, fact_b)
        assert rel_ab is not None
        assert rel_ab.relationship_type == "CORROBORATED"
        assert rel_ab.reasoning_method == "deterministic"

        # CASE 2: GENUINE CONTRADICTION (A vs C) - same period, conflicting values.
        rel_ac = relationship_between(fact_a, fact_c)
        assert rel_ac is not None
        assert rel_ac.relationship_type == "CONTRADICTED"
        assert rel_ac.reasoning_method == "llm"
        assert "FY2024" in rel_ac.explanation or "reporting period" in rel_ac.explanation.lower()

        # CASE 3: CONTEXTUALLY RECONCILED (C vs D) - same value-ish claim,
        # different fiscal year explains it.
        rel_cd = relationship_between(fact_c, fact_d)
        assert rel_cd is not None
        assert rel_cd.relationship_type == "CONTEXTUALLY_RECONCILED"
        assert rel_cd.contextual_dimensions.get("resolved_by") == "reporting_period"

        # CASE 4: EXTRACTION FAILURE (doc E) - the "fact" the mocked LLM
        # returned claimed evidence that is not actually present in the
        # source chunk. Validation must catch this, mark the fact invalid,
        # and record a first-class ExtractionFailure - never silently drop it.
        fact_e = session.query(Fact).filter(Fact.document_id == doc_e_id).one()
        assert fact_e.is_valid is False
        assert "not found" in fact_e.validation_notes

        failure = (
            session.query(ExtractionFailure)
            .filter(ExtractionFailure.document_id == doc_e_id, ExtractionFailure.stage == "validation")
            .first()
        )
        assert failure is not None
        assert failure.error_type == "unsupported_evidence"
        assert failure.final_status == "discarded"

"""
LangGraph orchestration (assignment section 3/34).

Each node has one job and reads/writes a shared, typed state dict. This
buys us: an explicit place to record per-stage errors, a clear retry
point (extract_facts loops per-chunk with its own failure handling
rather than aborting the whole document), and a processing trace that
maps directly onto the ProcessingRun.current_stage field shown in the
API/UI.

We deliberately do NOT put a node per trivial deterministic step - e.g.
hashing a file is a one-line function call inside ingest_document, not
its own graph node - per section 3's "do not create nodes merely to
claim LangGraph usage."
"""
from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from app.core.logging import get_logger
from app.db.database import get_session
from app.db.models import Chunk, Document, Fact, FactRelationship, ProcessingRun
from app.pipeline import ingestion
from app.pipeline.chunking import chunk_page
from app.pipeline.comparison import compare_facts
from app.pipeline.extraction import extract_facts_from_chunk
from app.pipeline.fact_builder import build_fact
from app.pipeline.failures import record_failure
from app.pipeline.retrieval import find_candidates
from app.vector_store.embeddings import embed_text

logger = get_logger("graph")


class PipelineState(TypedDict, total=False):
    document_id: str
    run_id: str
    file_path: str
    filename: str
    num_pages: int
    chunk_ids: list[str]
    fact_ids: list[str]
    relationships_created: int
    errors: list[str]
    stage: str


def _set_stage(run_id: str, stage: str) -> None:
    with get_session() as session:
        run = session.get(ProcessingRun, run_id)
        if run:
            run.current_stage = stage


def node_ingest_and_chunk(state: PipelineState) -> PipelineState:
    """ingest_document + extract_text + chunk_document + store_chunks, combined
    because they are all deterministic and always run together."""
    _set_stage(state["run_id"], "ingest_and_chunk")
    document_id = state["document_id"]
    chunk_ids: list[str] = []

    try:
        ingested = ingestion.ingest_pdf(state["file_path"])
    except ValueError as exc:
        with get_session() as session:
            doc = session.get(Document, document_id)
            if doc:
                doc.status = "failed"
                doc.error_message = str(exc)
            record_failure(
                session, stage="ingestion", error_type="pdf_parse_error", error_message=str(exc),
                run_id=state["run_id"], document_id=document_id, severity="high", final_status="unresolved",
                suggested_improvement="Verify the file is a valid, non-corrupted PDF.",
            )
        state["errors"] = state.get("errors", []) + [f"ingestion failed: {exc}"]
        state["num_pages"] = 0
        state["chunk_ids"] = []
        return state

    with get_session() as session:
        doc = session.get(Document, document_id)
        doc.num_pages = ingested.num_pages
        doc.status = "processing"

        if ingested.unextractable_pages:
            record_failure(
                session, stage="ingestion", error_type="empty_page",
                error_message=f"Pages with no extractable text: {ingested.unextractable_pages}",
                run_id=state["run_id"], document_id=document_id, severity="low",
                final_status="resolved",
                suggested_improvement="Add an OCR fallback for scanned/image-only pages.",
            )

        for page in ingested.pages:
            for cand in chunk_page(page):
                embedding = embed_text(cand.text)
                chunk = Chunk(
                    document_id=document_id,
                    page_number=cand.page_number,
                    chunk_index=cand.chunk_index,
                    section_header=cand.section_header,
                    text=cand.text,
                    char_start=cand.char_start,
                    char_end=cand.char_end,
                    embedding=embedding,
                )
                session.add(chunk)
                session.flush()
                chunk_ids.append(chunk.id)

    state["num_pages"] = ingested.num_pages
    state["chunk_ids"] = chunk_ids
    logger.info("Run %s: ingested %d pages -> %d chunks", state["run_id"], ingested.num_pages, len(chunk_ids))
    return state


def node_extract_facts(state: PipelineState) -> PipelineState:
    _set_stage(state["run_id"], "extract_facts")
    document_id = state["document_id"]
    fact_ids: list[str] = []

    with get_session() as session:
        doc = session.get(Document, document_id)
        filename = doc.filename
        num_pages = doc.num_pages

        for chunk_id in state.get("chunk_ids", []):
            chunk = session.get(Chunk, chunk_id)
            outcome = extract_facts_from_chunk(filename, chunk.page_number, chunk.text)

            if not outcome.ok:
                record_failure(
                    session, stage="extraction", error_type="llm_call_failed",
                    error_message=outcome.error or "unknown extraction error",
                    run_id=state["run_id"], document_id=document_id, chunk_id=chunk_id,
                    severity="medium", recovery_attempted=True, recovery_successful=False,
                    final_status="unresolved",
                    suggested_improvement="Retry with backoff, or fall back to a smaller/simpler prompt.",
                )
                continue

            for raw_fact in outcome.facts:
                built = build_fact(raw_fact, chunk.text, chunk.page_number, num_pages)

                if not built.is_valid:
                    record_failure(
                        session, stage="validation", error_type="unsupported_evidence",
                        error_message=(
                            f"Fact '{built.entity} {built.predicate} {built.object_text}' "
                            f"failed validation: {built.validation_notes}"
                        ),
                        run_id=state["run_id"], document_id=document_id, chunk_id=chunk_id,
                        severity="medium", recovery_attempted=True, recovery_successful=False,
                        final_status="discarded",
                        suggested_improvement=(
                            "Tighten the extraction prompt's instruction to quote verbatim "
                            "source text, or add a structured-repair pass that re-asks the "
                            "LLM to fix only the source_text field."
                        ),
                    )
                    # Still persisted (is_valid=False) so it's visible/inspectable,
                    # never silently dropped - but excluded from comparison.

                fact = Fact(
                    document_id=document_id,
                    chunk_id=chunk_id,
                    entity=built.entity,
                    predicate=built.predicate,
                    object_text=built.object_text,
                    fact_type=built.fact_type,
                    value_type=built.value_type,
                    normalized_value=built.normalized_value,
                    unit=built.unit,
                    currency=built.currency,
                    date=built.date,
                    start_date=built.start_date,
                    end_date=built.end_date,
                    reporting_period=built.reporting_period,
                    scope=built.scope,
                    location=built.location,
                    qualifiers=built.qualifiers,
                    source_text=built.source_text,
                    page_number=built.page_number,
                    char_start=built.char_start,
                    char_end=built.char_end,
                    confidence=built.confidence,
                    confidence_level=built.confidence_level,
                    extraction_method="llm",
                    is_valid=built.is_valid,
                    validation_notes=built.validation_notes,
                )
                session.add(fact)
                session.flush()
                fact_ids.append(fact.id)

        run = session.get(ProcessingRun, state["run_id"])
        run.facts_extracted = len(fact_ids)

    state["fact_ids"] = fact_ids
    logger.info("Run %s: extracted %d facts", state["run_id"], len(fact_ids))
    return state


def node_compare_facts(state: PipelineState) -> PipelineState:
    """retrieve_related_facts + compare_facts + analyze_context +
    classify_relationship + persist_relationships, combined: these all
    happen per-candidate-pair as one logical unit of work."""
    _set_stage(state["run_id"], "compare_facts")
    relationships_created = 0

    with get_session() as session:
        for fact_id in state.get("fact_ids", []):
            fact = session.get(Fact, fact_id)
            if fact is None or not fact.is_valid:
                continue

            candidates = find_candidates(session, fact)
            for candidate in candidates:
                other = candidate.fact

                existing = (
                    session.query(FactRelationship)
                    .filter(
                        ((FactRelationship.fact_a_id == fact.id) & (FactRelationship.fact_b_id == other.id))
                        | ((FactRelationship.fact_a_id == other.id) & (FactRelationship.fact_b_id == fact.id))
                    )
                    .first()
                )
                if existing:
                    continue

                try:
                    result = compare_facts(fact, other)
                except Exception as exc:  # noqa: BLE001
                    record_failure(
                        session, stage="comparison", error_type="comparison_exception",
                        error_message=str(exc), run_id=state["run_id"], document_id=state["document_id"],
                        fact_id=fact.id, severity="medium", final_status="unresolved",
                        suggested_improvement="Add stricter input validation before the comparison prompt.",
                    )
                    continue

                if not result.ok:
                    record_failure(
                        session, stage="comparison", error_type="llm_comparison_failed",
                        error_message=result.error or "unknown comparison error",
                        run_id=state["run_id"], document_id=state["document_id"], fact_id=fact.id,
                        severity="medium", recovery_attempted=True, recovery_successful=False,
                        final_status="unresolved",
                        suggested_improvement="Retry with backoff; fall back to UNCERTAIN with low confidence.",
                    )
                    continue

                if result.relationship_type == "UNRELATED":
                    continue

                rel = FactRelationship(
                    fact_a_id=fact.id,
                    fact_b_id=other.id,
                    relationship_type=result.relationship_type,
                    confidence=result.confidence,
                    confidence_level=(
                        "HIGH" if result.confidence >= 0.75 else "MEDIUM" if result.confidence >= 0.45 else "LOW"
                    ),
                    explanation=result.explanation,
                    contextual_dimensions=result.contextual_dimensions,
                    reasoning_method=result.reasoning_method,
                )
                session.add(rel)
                relationships_created += 1

        run = session.get(ProcessingRun, state["run_id"])
        run.relationships_created = relationships_created

    state["relationships_created"] = relationships_created
    logger.info("Run %s: created %d relationships", state["run_id"], relationships_created)
    return state


def node_finalize(state: PipelineState) -> PipelineState:
    _set_stage(state["run_id"], "completed")
    from datetime import datetime, timezone

    with get_session() as session:
        doc = session.get(Document, state["document_id"])
        run = session.get(ProcessingRun, state["run_id"])
        error_count = session.query(Fact).filter(
            Fact.document_id == state["document_id"], Fact.is_valid == False  # noqa: E712
        ).count()

        if state.get("errors") and state.get("num_pages", 0) == 0:
            doc.status = "failed"
            run.status = "failed"
        else:
            doc.status = "completed"
            run.status = "completed"
        run.error_count = error_count
        run.completed_at = datetime.now(timezone.utc)

    return state


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("ingest_and_chunk", node_ingest_and_chunk)
    graph.add_node("extract_facts", node_extract_facts)
    graph.add_node("compare_facts", node_compare_facts)
    graph.add_node("finalize", node_finalize)

    graph.set_entry_point("ingest_and_chunk")

    def _route_after_ingest(state: PipelineState) -> str:
        if state.get("errors") and state.get("num_pages", 0) == 0:
            return "finalize"
        return "extract_facts"

    graph.add_conditional_edges("ingest_and_chunk", _route_after_ingest, {
        "extract_facts": "extract_facts", "finalize": "finalize",
    })
    graph.add_edge("extract_facts", "compare_facts")
    graph.add_edge("compare_facts", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


_compiled_graph = None


def get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_pipeline(document_id: str, run_id: str, file_path: str) -> PipelineState:
    graph = get_compiled_graph()
    initial_state: PipelineState = {
        "document_id": document_id,
        "run_id": run_id,
        "file_path": file_path,
        "errors": [],
    }
    final_state = graph.invoke(initial_state)
    return final_state

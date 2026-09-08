from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.api.serializers import fact_to_schema, relationship_to_schema
from app.core.config import settings
from app.core.logging import get_logger
from app.db.database import get_session, init_db, session_dependency
from app.db.models import Document, ExtractionFailure, Fact, FactRelationship, ProcessingRun
from app.models.schemas import (
    DocumentOut,
    FactOut,
    FailureOut,
    ProcessingRunOut,
    RelationshipOut,
    SearchResult,
    UploadResponse,
)
from app.pipeline.graph import run_pipeline
from app.pipeline.ingestion import hash_file
from app.vector_store.store import find_similar_facts

logger = get_logger("api")

app = FastAPI(
    title="Superjoin Fact Knowledge Layer",
    description="Extracts grounded, cross-document facts from PDFs and classifies "
    "corroboration / contradiction / contextual reconciliation between them.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    logger.info("Database initialized at %s", settings.database_url)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "llm_configured": bool(settings.gemini_api_key)}


def _run_pipeline_safely(document_id: str, run_id: str, file_path: str) -> None:
    try:
        run_pipeline(document_id, run_id, file_path)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Pipeline crashed for document %s", document_id)
        with get_session() as session:
            doc = session.get(Document, document_id)
            if doc:
                doc.status = "failed"
                doc.error_message = str(exc)
            run = session.get(ProcessingRun, run_id)
            if run:
                run.status = "failed"


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile, background_tasks: BackgroundTasks) -> UploadResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only .pdf files are supported")

    dest_dir = settings.uploads_dir
    dest_path = dest_dir / file.filename
    # avoid clobbering an existing upload with the same name
    counter = 1
    while dest_path.exists():
        stem = Path(file.filename).stem
        dest_path = dest_dir / f"{stem}_{counter}.pdf"
        counter += 1

    with open(dest_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    file_hash = hash_file(str(dest_path))

    with get_session() as session:
        existing = session.query(Document).filter(Document.file_hash == file_hash).first()
        if existing:
            dest_path.unlink(missing_ok=True)
            run = session.query(ProcessingRun).filter(ProcessingRun.document_id == existing.id).first()
            logger.info("Duplicate upload detected (hash match) -> returning existing document %s", existing.id)
            return UploadResponse(
                document=DocumentOut.model_validate(existing),
                run=ProcessingRunOut.model_validate(run) if run else _empty_run(existing.id),
            )

        doc = Document(
            filename=file.filename,
            file_hash=file_hash,
            stored_path=str(dest_path),
            status="uploaded",
        )
        session.add(doc)
        session.flush()

        run = ProcessingRun(document_id=doc.id, status="pending", current_stage="queued")
        session.add(run)
        session.flush()

        document_id, run_id = doc.id, run.id
        document_out = DocumentOut.model_validate(doc)
        run_out = ProcessingRunOut.model_validate(run)

    # Processed in the background so a large PDF (many sequential LLM
    # calls for extraction + comparison) doesn't block the HTTP response.
    # The client polls GET /documents/{id}/runs for status/progress.
    background_tasks.add_task(_run_pipeline_safely, document_id, run_id, str(dest_path))

    return UploadResponse(document=document_out, run=run_out)


def _empty_run(document_id: str) -> ProcessingRunOut:
    from datetime import datetime, timezone

    return ProcessingRunOut(
        id="none", document_id=document_id, status="completed", current_stage=None,
        facts_extracted=0, relationships_created=0, error_count=0,
        started_at=datetime.now(timezone.utc), completed_at=datetime.now(timezone.utc),
    )


@app.get("/documents", response_model=list[DocumentOut])
def list_documents(session: Session = Depends(session_dependency)) -> list[DocumentOut]:
    docs = session.query(Document).order_by(Document.created_at.desc()).all()
    return [DocumentOut.model_validate(d) for d in docs]


@app.get("/documents/{document_id}", response_model=DocumentOut)
def get_document(document_id: str, session: Session = Depends(session_dependency)) -> DocumentOut:
    doc = session.get(Document, document_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    return DocumentOut.model_validate(doc)


@app.get("/documents/{document_id}/facts", response_model=list[FactOut])
def get_document_facts(document_id: str, session: Session = Depends(session_dependency)) -> list[FactOut]:
    doc = session.get(Document, document_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    facts = session.query(Fact).filter(Fact.document_id == document_id).all()
    return [fact_to_schema(f) for f in facts]


@app.get("/documents/{document_id}/runs", response_model=list[ProcessingRunOut])
def get_document_runs(document_id: str, session: Session = Depends(session_dependency)) -> list[ProcessingRunOut]:
    runs = (
        session.query(ProcessingRun)
        .filter(ProcessingRun.document_id == document_id)
        .order_by(ProcessingRun.started_at.desc())
        .all()
    )
    return [ProcessingRunOut.model_validate(r) for r in runs]


@app.get("/facts", response_model=list[FactOut])
def list_facts(
    entity: str | None = None,
    predicate: str | None = None,
    only_valid: bool = True,
    session: Session = Depends(session_dependency),
) -> list[FactOut]:
    q = session.query(Fact)
    if entity:
        q = q.filter(Fact.entity.ilike(f"%{entity}%"))
    if predicate:
        q = q.filter(Fact.predicate.ilike(f"%{predicate}%"))
    if only_valid:
        q = q.filter(Fact.is_valid == True)  # noqa: E712
    return [fact_to_schema(f) for f in q.limit(500).all()]


@app.get("/facts/{fact_id}", response_model=FactOut)
def get_fact(fact_id: str, session: Session = Depends(session_dependency)) -> FactOut:
    fact = session.get(Fact, fact_id)
    if not fact:
        raise HTTPException(404, "Fact not found")
    return fact_to_schema(fact)


@app.get("/facts/{fact_id}/relationships", response_model=list[RelationshipOut])
def get_fact_relationships(fact_id: str, session: Session = Depends(session_dependency)) -> list[RelationshipOut]:
    fact = session.get(Fact, fact_id)
    if not fact:
        raise HTTPException(404, "Fact not found")
    rels = (
        session.query(FactRelationship)
        .filter((FactRelationship.fact_a_id == fact_id) | (FactRelationship.fact_b_id == fact_id))
        .all()
    )
    out = []
    for rel in rels:
        fact_a = session.get(Fact, rel.fact_a_id)
        fact_b = session.get(Fact, rel.fact_b_id)
        out.append(relationship_to_schema(rel, fact_a, fact_b))
    return out


@app.get("/relationships", response_model=list[RelationshipOut])
def list_relationships(
    relationship_type: str | None = None,
    session: Session = Depends(session_dependency),
) -> list[RelationshipOut]:
    q = session.query(FactRelationship)
    if relationship_type:
        q = q.filter(FactRelationship.relationship_type == relationship_type.upper())
    rels = q.order_by(FactRelationship.created_at.desc()).limit(500).all()
    out = []
    for rel in rels:
        fact_a = session.get(Fact, rel.fact_a_id)
        fact_b = session.get(Fact, rel.fact_b_id)
        if fact_a is None or fact_b is None:
            continue
        out.append(relationship_to_schema(rel, fact_a, fact_b))
    return out


@app.get("/failures", response_model=list[FailureOut])
def list_failures(session: Session = Depends(session_dependency)) -> list[FailureOut]:
    failures = session.query(ExtractionFailure).order_by(ExtractionFailure.created_at.desc()).limit(500).all()
    return [FailureOut.model_validate(f) for f in failures]


@app.get("/search", response_model=list[SearchResult])
def search_facts(q: str, top_k: int = 10, session: Session = Depends(session_dependency)) -> list[SearchResult]:
    if not q.strip():
        return []
    scored = find_similar_facts(session, q, top_k=top_k)
    return [SearchResult(fact=fact_to_schema(s.fact), score=s.score) for s in scored]

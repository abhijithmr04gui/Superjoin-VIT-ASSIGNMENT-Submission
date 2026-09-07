from __future__ import annotations

from app.db.database import get_session
from app.db.models import Document, ProcessingRun
from app.pipeline.graph import run_pipeline


def process_pdf(path: str, filename: str) -> tuple[str, str]:
    """Create Document + ProcessingRun rows and run the pipeline synchronously,
    the same way the /documents/upload endpoint does. Returns (document_id, run_id)."""
    from app.pipeline.ingestion import hash_file

    with get_session() as session:
        doc = Document(filename=filename, file_hash=hash_file(path), stored_path=path, status="uploaded")
        session.add(doc)
        session.flush()
        run = ProcessingRun(document_id=doc.id, status="running", current_stage="queued")
        session.add(run)
        session.flush()
        document_id, run_id = doc.id, run.id

    run_pipeline(document_id, run_id, path)
    return document_id, run_id

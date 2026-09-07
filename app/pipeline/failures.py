from __future__ import annotations

from sqlalchemy.orm import Session

from app.db.models import ExtractionFailure
from app.core.logging import get_logger

logger = get_logger("failures")


def record_failure(
    session: Session,
    stage: str,
    error_type: str,
    error_message: str,
    run_id: str | None = None,
    document_id: str | None = None,
    chunk_id: str | None = None,
    fact_id: str | None = None,
    severity: str = "medium",
    recovery_attempted: bool = False,
    recovery_successful: bool = False,
    final_status: str = "unresolved",
    suggested_improvement: str | None = None,
) -> ExtractionFailure:
    failure = ExtractionFailure(
        run_id=run_id,
        document_id=document_id,
        chunk_id=chunk_id,
        fact_id=fact_id,
        stage=stage,
        error_type=error_type,
        error_message=error_message[:4000],
        severity=severity,
        recovery_attempted=recovery_attempted,
        recovery_successful=recovery_successful,
        final_status=final_status,
        suggested_improvement=suggested_improvement,
    )
    session.add(failure)
    session.flush()
    logger.warning(
        "Recorded failure [%s/%s] stage=%s doc=%s: %s",
        severity, final_status, stage, document_id, error_message[:200],
    )
    return failure

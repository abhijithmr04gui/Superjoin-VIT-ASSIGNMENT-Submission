"""
Document ingestion. Deterministic, no LLM calls here.

Nothing in this module knows about "the starter PDFs" - it accepts any
PDF path and returns page-level text plus metadata. If a page yields no
extractable text (e.g. a scanned image page with no OCR layer), that is
recorded explicitly rather than silently skipped.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import fitz  # PyMuPDF

from app.core.logging import get_logger

logger = get_logger("ingestion")


@dataclass
class PageText:
    page_number: int  # 1-indexed
    text: str
    is_empty: bool


@dataclass
class IngestedDocument:
    file_hash: str
    num_pages: int
    pages: list[PageText] = field(default_factory=list)
    unextractable_pages: list[int] = field(default_factory=list)


def hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def ingest_pdf(path: str) -> IngestedDocument:
    """
    Extract page-by-page text from an arbitrary PDF. Raises ValueError if
    the file cannot be opened as a PDF at all (corrupt file, wrong format).
    """
    file_hash = hash_file(path)

    try:
        doc = fitz.open(path)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Could not open file as PDF: {exc}") from exc

    pages: list[PageText] = []
    unextractable: list[int] = []

    for i in range(len(doc)):
        page = doc[i]
        # "text" mode preserves reading order reasonably well and is fast.
        # Tables are not structurally preserved here (see README limitations);
        # we fall back to a best-effort layout-aware extraction as well.
        text = page.get_text("text") or ""
        text = text.strip()

        if not text:
            unextractable.append(i + 1)
            logger.info("Page %d has no extractable text (possibly scanned/image-only)", i + 1)

        pages.append(PageText(page_number=i + 1, text=text, is_empty=not bool(text)))

    doc.close()

    result = IngestedDocument(
        file_hash=file_hash,
        num_pages=len(pages),
        pages=pages,
        unextractable_pages=unextractable,
    )
    logger.info(
        "Ingested PDF: %d pages, %d unextractable", result.num_pages, len(unextractable)
    )
    return result

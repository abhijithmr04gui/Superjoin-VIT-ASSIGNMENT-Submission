"""
Chunking strategy: page-bounded, overlapping character windows.

Rationale (documented in README too): chunking per-page rather than
across the whole document keeps every chunk attributable to exactly one
page number, which evidence linking depends on. Within a page, we use
overlapping windows so a fact whose sentence sits near a chunk boundary
still has full context in at least one chunk. We split on paragraph/
sentence boundaries where possible instead of cutting mid-sentence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.config import settings
from app.pipeline.ingestion import PageText

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class ChunkCandidate:
    page_number: int
    chunk_index: int
    text: str
    char_start: int
    char_end: int
    section_header: str | None = None


def _detect_header(text: str) -> str | None:
    """Best-effort: a short all-caps or title-cased first line often is a header."""
    first_line = text.strip().split("\n", 1)[0].strip()
    if not first_line or len(first_line) > 80:
        return None
    words = first_line.split()
    if not words:
        return None
    caps_ratio = sum(1 for w in words if w[:1].isupper()) / len(words)
    if caps_ratio >= 0.7 and len(first_line.split()) <= 12:
        return first_line
    return None


def chunk_page(
    page: PageText,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[ChunkCandidate]:
    if page.is_empty:
        return []

    chunk_size = chunk_size or settings.chunk_size_chars
    overlap = overlap if overlap is not None else settings.chunk_overlap_chars

    text = page.text
    header = _detect_header(text)

    if len(text) <= chunk_size:
        return [
            ChunkCandidate(
                page_number=page.page_number,
                chunk_index=0,
                text=text,
                char_start=0,
                char_end=len(text),
                section_header=header,
            )
        ]

    sentences = _SENTENCE_SPLIT.split(text)
    chunks: list[ChunkCandidate] = []
    cursor = 0
    current = ""
    current_start = 0
    idx = 0

    def flush(end_pos: int) -> None:
        nonlocal current, current_start, idx
        if current.strip():
            chunks.append(
                ChunkCandidate(
                    page_number=page.page_number,
                    chunk_index=idx,
                    text=current.strip(),
                    char_start=current_start,
                    char_end=end_pos,
                    section_header=header if idx == 0 else None,
                )
            )
            idx += 1

    pos = 0
    for sentence in sentences:
        sent_len = len(sentence)
        if len(current) + sent_len > chunk_size and current:
            end_pos = pos
            flush(end_pos)
            # start new chunk with overlap: keep tail of previous chunk
            overlap_text = current[-overlap:] if overlap > 0 else ""
            current = overlap_text + " " + sentence if overlap_text else sentence
            current_start = max(pos - len(overlap_text), 0)
        else:
            if not current:
                current_start = pos
            current = f"{current} {sentence}".strip() if current else sentence
        pos += sent_len + 1

    flush(len(text))

    return chunks

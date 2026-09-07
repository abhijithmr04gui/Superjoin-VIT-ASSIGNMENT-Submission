# Superjoin VIT 2026 — PDF Fact Knowledge Layer

Extracts grounded, evidence-linked facts from arbitrary PDFs and classifies
relationships between facts found across different documents:
**corroboration**, **contradiction**, or **contextual reconciliation** —
with explicit confidence and explicit failure handling.

> "Given a set of PDFs, the system extracts structured facts, preserves
> their original evidence, retrieves potentially related facts, compares
> their semantic and contextual dimensions, and determines whether they
> corroborate, contradict, or can be reconciled — with explicit
> uncertainty and failure handling." This is a Fact Knowledge Layer, not
> a PDF chatbot.

---

## Video Demo

`[ADD LINK HERE — 3 minutes or less, see "Suggested demo script" below]`

## Setup and Run Instructions

### Prerequisites

- Python 3.11+
- An Anthropic API key (used for fact extraction, entity resolution, and
  fact comparison)

### Installation

```bash
git clone <this-repo>
cd superjoin-vit-2026
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Environment variables

```bash
cp .env.example .env
# then edit .env and set ANTHROPIC_API_KEY
```

See `.env.example` for every variable and its default. Nothing else
needs configuring — the "vector database" and structured store are both
local files under `./data/` created automatically on first run.

### Start the backend

```bash
uvicorn app.api.main:app --reload
# API docs: http://localhost:8000/docs
```

### Start the frontend (optional, separate terminal)

```bash
streamlit run frontend/app.py
```

### Upload PDFs

Either drag a PDF into the Streamlit "Upload & Documents" tab, or:

```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@data/starter-dataset/delhivery/03-delhivery-q4-fy24-earnings-presentation.pdf"
```

Upload is synchronous in this prototype — the response only returns once
ingestion, extraction, normalization, and comparison have all finished
for that document (see Trade-offs).

### Run tests

```bash
pytest tests/ -v
```

All tests run with the LLM mocked out (no API key or network access
needed) except the pipeline's real network call path itself, which is
only exercised when you actually hit `/documents/upload` with a live key.

---

## Starter dataset

`data/starter-dataset/` contains two independent sets of real PDFs used
during development and testing (see each folder's own README for exact
provenance/curation notes):

- `delhivery/` — a 2022 IPO prospectus, the FY24 annual report, and the
  Q4 FY24 earnings presentation for Delhivery Limited. These three
  overlap heavily on FY24 revenue/EBITDA figures reported at different
  times, which is what naturally produces corroboration, contradiction,
  and time-based reconciliation cases without any hand-tuning.
- `india-macroeconomy/` — the Economic Survey 2024-25, the RBI Annual
  Report 2024-25, and the IMF's 2025 India Article IV excerpt. This set
  was never looked at while writing the extraction/comparison prompts or
  the fact schema, so running it through the same pipeline with zero code
  changes is the project's own "new PDF" generalization check.

No filename, entity, or fact from either set is hard-coded anywhere in
`app/`.

---

## Approach

### Architecture

```
PDF upload
  -> ingestion (PyMuPDF, page-by-page text + hash + empty-page detection)
  -> chunking  (page-bounded, overlapping, sentence-aware)
  -> [LangGraph pipeline]
       ingest_and_chunk -> extract_facts -> compare_facts -> finalize
  -> FastAPI (upload/documents/facts/relationships/search/failures)
  -> Streamlit UI (evidence, relationships, explanations, confidence)
```

### Fact schema

One generic `Fact` table (`app/db/models.py`) with a subject/predicate/
object-ish core (`entity`, `predicate`, `object_text`) plus optional
structured fields (`unit`, `currency`, `date`, `reporting_period`,
`scope`, `qualifiers`, `normalized_value`). New fact types (a new
predicate, a new kind of qualifier) need **zero schema changes** — there
is no `revenue_facts` / `director_facts` table split.

Every `Fact` row carries its own evidence inline: `source_text`,
`page_number`, and (when an exact match is found) character offsets. A
fact is never persisted without evidence; the one exception is a fact
that *fails* evidence validation, which is still persisted with
`is_valid=False` so it stays inspectable rather than silently vanishing.

### Ingestion & chunking

PyMuPDF extracts text page-by-page (`app/pipeline/ingestion.py`). Pages
with no extractable text (scanned/image-only pages) are recorded, not
skipped silently — this happened for real on page 1 of the IMF excerpt
in the starter dataset (a cover page) during testing.

Chunking (`app/pipeline/chunking.py`) is page-bounded and
sentence-aware with configurable overlap (`CHUNK_SIZE_CHARS` /
`CHUNK_OVERLAP_CHARS`), so every chunk maps to exactly one page number
for evidence purposes, while overlap keeps sentences near a chunk
boundary fully readable in at least one chunk.

### Extraction, validation, normalization

Three separate stages, deliberately:

1. **Extraction** (`app/pipeline/extraction.py`) — one focused LLM
   prompt per chunk (`prompts/fact_extraction.py`) returning structured
   JSON. Malformed items are dropped, not coerced.
2. **Validation** (`app/pipeline/validation.py`) — deterministic, no
   LLM. The important check: does the claimed `source_text` actually
   appear (exactly, or with high textual containment) inside the chunk
   it was supposedly extracted from? This is the guard against
   hallucinated evidence.
3. **Normalization** (`app/pipeline/normalization.py`) — deterministic
   parsing of numbers ("$10 million" / "USD 10M" / "₹4.2 crore" / "15%"),
   dates ("FY2024" / "Q3 2024" / "January 2025"), and entity name
   canonicalization. Pure functions, unit-tested in isolation
   (`tests/test_normalization.py`).

### Retrieval & comparison

Candidate retrieval (`app/pipeline/retrieval.py`) narrows the comparison
set instead of comparing every fact against every other fact: a
candidate must match on entity, predicate/fact_type, *or* vector
similarity — vector similarity alone is never sufficient (see Vector
database, below).

Comparison (`app/pipeline/comparison.py`) tries a **deterministic
shortcut first**: if entity, predicate, period, scope, and qualifiers
all match exactly *and* normalized values agree, it's classified
`CORROBORATED` without an LLM call. Critically, the deterministic path
**never** returns `CONTRADICTED` — any value disagreement, even with
everything else matching, is deferred to the LLM comparison prompt
(`prompts/fact_comparison.py`), which walks the ten-step context
hierarchy from the assignment brief (entity → predicate → units → time →
scope → qualifiers → values → contextual explanation → contradiction →
uncertainty) before choosing one of `CORROBORATED` / `CONTRADICTED` /
`LIKELY_CONTRADICTION` / `CONTEXTUALLY_RECONCILED` / `UNCERTAIN` /
`UNRELATED`.

### LangGraph usage

`app/pipeline/graph.py` wires four nodes: `ingest_and_chunk` →
`extract_facts` → `compare_facts` → `finalize`, with a conditional edge
that skips straight to `finalize` if ingestion itself fails (corrupt
PDF). Nodes were kept coarse deliberately — a node per trivial
deterministic step (e.g. hashing a file) would exist only to claim
LangGraph usage, which the assignment explicitly discourages. Each node
does own its failure handling and updates `ProcessingRun.current_stage`,
which the API/UI surface directly.

### Vector database

Candidate generation only (never the final relationship decision). Uses
`sklearn.feature_extraction.text.HashingVectorizer` instead of a
transformer embedding model — see Trade-offs for why.

### Confidence & failure handling

Every `Fact` and `FactRelationship` carries a numeric `confidence` and a
bucketed `confidence_level` (HIGH ≥0.75, MEDIUM ≥0.45, else LOW).
Validation can only ever *lower* an LLM's self-reported confidence, never
raise it. `ExtractionFailure` (`app/db/models.py`) is a first-class table,
not a log line — every stage (ingestion, extraction, validation,
comparison) records failures there with a stage, error type, severity,
recovery attempt outcome, and a suggested improvement, and the
`/failures` endpoint plus the Streamlit "Failures / Health" tab surface
them directly. Nothing is hidden to make a demo look cleaner.

---

## Trade-offs

- **Anthropic Claude, not a mix of providers** — the assignment only
  needs one capable LLM behind focused prompts; adding a second provider
  would add integration surface without adding capability here.
- **HashingVectorizer instead of a transformer embedding model** — it is
  stateless (no corpus-fit step), so adding document N+1 never requires
  re-embedding documents 1..N — genuinely incremental. It is also
  dependency-light (no GPU/model download) and fast everywhere. The
  cost: it captures lexical/n-gram overlap, not deep paraphrase
  similarity. This is an acceptable prototype trade-off *because* vector
  similarity here is only a candidate-generation signal (never the final
  corroboration/contradiction decision), and the paraphrase reasoning
  that actually matters ("$10 million" vs "USD 10M") is handled by
  deterministic normalization plus the LLM comparison stage, not by the
  embedding. Documented upgrade path: swap `app/vector_store/embeddings.py`
  for `sentence-transformers` or a hosted embeddings API — nothing else
  in the pipeline depends on the embedding implementation.
- **SQLite-stored vectors + in-process cosine similarity, not
  Chroma/FAISS/pgvector** — the assignment itself prefers "a simple
  locally runnable option unless another choice has a strong engineering
  justification" for a prototype this size. `app/vector_store/store.py`
  isolates this decision behind two functions so swapping in a real
  vector DB later is a contained change.
- **Synchronous upload processing, no background job queue** — simplest
  correct behavior for a prototype and for a 3-minute demo (the response
  itself shows facts/relationships extracted, no polling needed). Cost:
  a very large PDF or a burst of uploads would block the request thread.
  Documented next step: move `run_pipeline` to a background task/queue
  (Celery/RQ or FastAPI `BackgroundTasks` + a `GET` status endpoint,
  which already exists via `ProcessingRun`).
- **Deterministic corroboration shortcut only ever fires on an exact
  match across every dimension** — intentionally conservative. It saves
  an LLM call in the easy case (see assignment section 30) without ever
  letting ad hoc code make a contradiction/reconciliation judgment call,
  which the assignment explicitly reserves for LLM reasoning (section 16).
- **LLM comparison confidence is not statistically calibrated** — it is
  the model's self-reported confidence, used only as a relative signal
  (HIGH/MEDIUM/LOW bucketing) and explicitly documented as such rather
  than presented as a calibrated probability.

---

## Limitations and Next Steps

- **Chart-heavy slide decks lose visual context.** Discovered on the
  starter dataset itself: page 15 of the Delhivery Q4 FY24 earnings
  presentation renders a bar+bubble chart ("Net Working Capital Days")
  whose axis labels and data values extract as a flat, disconnected
  sequence of bare numbers (`73 47 37 ... 106 87 74 77 66 33 40 37 39
  35 ... 38 31`) with no structural link back to which number belongs to
  which year or series. PyMuPDF's text-mode extraction reads visual
  position, not chart semantics, so a number's meaning that depended on
  its position relative to a bar or bubble is lost. The extraction
  prompt's instruction not to force every fragment into a fact catches
  most of this (bare number sequences get skipped rather than turned
  into fabricated facts), but a real, checkable number sitting inside a
  chart can be missed entirely. Next step: a dedicated table/chart
  extraction pass (e.g. `pdfplumber`'s table detection, or a
  vision-model pass over rasterized chart regions) feeding the same
  extraction prompt with reconstructed row/series context.
- **No OCR fallback.** A scanned/image-only page (this happened for real
  on page 1 of the IMF excerpt in the starter dataset) is detected and
  recorded, but its content is not recovered. Next step: Tesseract or a
  vision-model OCR pass as a fallback path in `ingestion.py`.
- **Entity resolution is name-based, not LLM-verified by default.**
  `normalize_entity_name` (suffix-stripping + lowercasing) is used for
  the deterministic corroboration shortcut and for candidate retrieval.
  The LLM-backed entity resolution prompt (`prompts/entity_resolution.py`)
  exists but is not yet wired into the comparison node — right now the
  comparison LLM call is given both raw entity strings and asked to
  account for naming variation itself as part of the wider comparison
  prompt. A dedicated resolution pass would help most on datasets with
  many differently-abbreviated entities.
- **Table structure is not explicitly preserved.** Dense financial
  tables (see the Delhivery annual report / RBI appendix tables in the
  starter dataset) extract as text with reasonable reading order but
  without row/column structure, which increases the risk of a numeric
  fact losing its row label context in a very dense table. A
  table-detection pass (e.g. `pdfplumber`) is the natural next step.
- **Synchronous processing** (see Trade-offs above) — no background job
  queue yet, so very large PDFs or many concurrent uploads will block.
- **Vector retrieval quality** (see Trade-offs above) is lexical, not
  semantic. On facts that share almost no vocabulary despite being about
  the same underlying thing, candidate retrieval could miss them if
  entity/predicate also happen to differ (rare given normalization, but
  possible with typos or unseen abbreviations).
- **Incremental processing exists at the data-model level but is not
  yet fully optimized.** A new upload's facts are already compared only
  against existing facts (not re-comparing all existing pairs against
  each other), and its chunks are embedded independently (no corpus
  refit needed). What is not yet implemented: skipping re-embedding of
  unchanged chunks in general, and a metadata-filtered candidate index
  for datasets with many more documents than the starter set.

---

## Additional Notes

- `app/pipeline/comparison.py::deterministic_precheck` is unit-tested
  specifically to assert it never returns `CONTRADICTED` on its own
  (`tests/test_comparison.py`) — this is a load-bearing invariant for
  the "don't overtrust ad hoc code with a contradiction verdict"
  requirement, so it is tested directly rather than only indirectly via
  the end-to-end pipeline test.
- `tests/test_pipeline_integration.py` runs the full LangGraph pipeline
  (real chunking, real validation, real normalization, real candidate
  retrieval, real persistence) with only the two LLM call sites mocked,
  and asserts all four required demonstration cases directly against the
  database: corroboration, contradiction, contextual reconciliation, and
  a caught extraction failure (simulated hallucinated evidence).
- Duplicate uploads are detected by file hash (`Document.file_hash`) and
  short-circuit to the existing document rather than reprocessing.

---

## Suggested demo script (≤ 3 minutes)

1. **0:00–0:20** — State the problem: facts scattered across PDFs, worded
   differently, sometimes conflicting, sometimes only apparently
   conflicting.
2. **0:20–0:45** — Upload the Delhivery prospectus, annual report, and
   earnings presentation from `data/starter-dataset/delhivery/`.
3. **0:45–1:15** — Open a FY24 revenue fact, show its evidence (page +
   quoted source text).
4. **1:15–1:40** — Show the corroborated relationship between two
   independently worded revenue statements.
5. **1:40–2:00** — Show the contradiction case (two figures for what
   looks like the same period).
6. **2:00–2:20** — Show the reconciliation case (different fiscal years
   explain a value difference).
7. **2:20–2:40** — Open the Failures tab: the Net Working Capital chart
   extraction ambiguity, or a caught hallucinated-evidence case, with the
   suggested improvement shown.
8. **2:40–3:00** — One sentence on architecture (LangGraph pipeline,
   local vector store, deterministic shortcut + LLM reasoning) and the
   top limitation (chart/table structure loss).

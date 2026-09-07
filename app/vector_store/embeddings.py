"""
Embedding function used purely for CANDIDATE GENERATION (never for the
final corroboration/contradiction decision - see comparison.py and
assignment section 4).

Engineering trade-off (documented in README): we use scikit-learn's
HashingVectorizer instead of a transformer embedding model. It is:
  - stateless (no corpus fit step), so adding document #101 never requires
    re-embedding documents #1-100 - genuinely incremental;
  - dependency-light and fast to run anywhere, no model download/GPU;
  - combined with TF-IDF-style term weighting via a two-stage transform.

Trade-off: it captures lexical/term overlap, not deep paraphrase
similarity. Because vector similarity here is only used to narrow the
candidate set (assignment section 4 & 13), and the real corroboration
paraphrase reasoning ("$10 million" vs "10M") happens in normalization +
the LLM comparison stage, this is an acceptable prototype trade-off. A
transformer model (sentence-transformers, OpenAI/Voyage embeddings) is
the documented upgrade path for production.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

_VECTOR_DIM = 512

_vectorizer = HashingVectorizer(
    n_features=_VECTOR_DIM,
    alternate_sign=False,
    norm="l2",
    ngram_range=(1, 2),
    stop_words="english",
)


def embed_text(text: str) -> list[float]:
    if not text or not text.strip():
        return [0.0] * _VECTOR_DIM
    vec = _vectorizer.transform([text])
    return vec.toarray()[0].tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    vecs = _vectorizer.transform(texts)
    return vecs.toarray().tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)

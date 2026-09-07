from app.db.models import Fact
from app.pipeline.comparison import deterministic_precheck


def _make_fact(**kwargs) -> Fact:
    base = dict(
        document_id="doc1",
        entity="Example Corp",
        predicate="revenue",
        object_text="$10 million",
        fact_type="numeric",
        value_type="number",
        normalized_value={"number": 10_000_000},
        reporting_period="FY2024",
        scope=None,
        qualifiers=[],
        source_text="revenue was $10 million",
        page_number=1,
        confidence=0.8,
    )
    base.update(kwargs)
    return Fact(**base)


def test_deterministic_precheck_exact_match_corroborates():
    a = _make_fact()
    b = _make_fact(document_id="doc2", object_text="USD 10M", source_text="USD 10M revenue")
    result = deterministic_precheck(a, b)
    assert result is not None
    assert result.relationship_type == "CORROBORATED"
    assert result.reasoning_method == "deterministic"


def test_deterministic_precheck_different_period_defers_to_llm():
    a = _make_fact(reporting_period="FY2024")
    b = _make_fact(document_id="doc2", reporting_period="FY2025", normalized_value={"number": 15_000_000})
    result = deterministic_precheck(a, b)
    assert result is None  # left for the LLM stage - not a deterministic call


def test_deterministic_precheck_different_values_never_returns_contradiction():
    a = _make_fact(normalized_value={"number": 10_000_000})
    b = _make_fact(document_id="doc2", normalized_value={"number": 15_000_000})
    result = deterministic_precheck(a, b)
    # Same entity/predicate/period/scope but disagreeing values: the
    # deterministic shortcut must never itself call this a contradiction.
    assert result is None


def test_deterministic_precheck_different_entity_defers():
    a = _make_fact(entity="Example Corp")
    b = _make_fact(document_id="doc2", entity="Other Corp")
    assert deterministic_precheck(a, b) is None

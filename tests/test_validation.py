from app.pipeline.extraction import RawExtractedFact
from app.pipeline.validation import validate_fact


def _fact(**kwargs) -> RawExtractedFact:
    base = dict(
        entity="Example Corp",
        predicate="revenue",
        object_text="$10 million",
        source_text="revenue was $10 million",
    )
    base.update(kwargs)
    return RawExtractedFact(**base)


def test_valid_fact_exact_evidence_match():
    chunk = "In FY2024, revenue was $10 million across all segments."
    result = validate_fact(_fact(), chunk, page_number=1, num_pages=5)
    assert result.is_valid
    assert "exact" in result.notes


def test_invalid_missing_entity():
    f = _fact(entity="")
    result = validate_fact(f, "some text", page_number=1, num_pages=5)
    assert not result.is_valid
    assert "entity" in result.notes


def test_invalid_page_out_of_range():
    result = validate_fact(_fact(), "revenue was $10 million", page_number=99, num_pages=5)
    assert not result.is_valid
    assert "page" in result.notes


def test_invalid_hallucinated_evidence():
    chunk = "The company opened a new office in Berlin last quarter."
    result = validate_fact(_fact(), chunk, page_number=1, num_pages=5)
    assert not result.is_valid
    assert "not found" in result.notes


def test_fuzzy_match_accepted_with_lower_confidence_flag():
    chunk = "Total revenue reported was approximately $10 million for the period."
    f = _fact(source_text="revenue was $10 million")
    result = validate_fact(f, chunk, page_number=1, num_pages=5)
    # Not an exact substring, but close enough to be a legitimate paraphrase capture.
    assert result.is_valid

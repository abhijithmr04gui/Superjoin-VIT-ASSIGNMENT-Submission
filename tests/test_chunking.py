from app.pipeline.chunking import chunk_page
from app.pipeline.ingestion import PageText


def test_short_page_single_chunk():
    page = PageText(page_number=1, text="Revenue was $10 million in FY2024.", is_empty=False)
    chunks = chunk_page(page, chunk_size=1000, overlap=100)
    assert len(chunks) == 1
    assert chunks[0].page_number == 1
    assert "10 million" in chunks[0].text


def test_empty_page_no_chunks():
    page = PageText(page_number=2, text="", is_empty=True)
    assert chunk_page(page) == []


def test_long_page_multiple_chunks_with_overlap():
    sentence = "This is a filler sentence about the company and its operations. "
    text = sentence * 40  # long enough to force multiple chunks
    page = PageText(page_number=3, text=text, is_empty=False)
    chunks = chunk_page(page, chunk_size=300, overlap=50)
    assert len(chunks) > 1
    for c in chunks:
        assert c.page_number == 3
    # chunk_index should be sequential starting at 0
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

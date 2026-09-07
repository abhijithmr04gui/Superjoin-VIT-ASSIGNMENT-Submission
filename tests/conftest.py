"""
IMPORTANT: environment variables must be set before `app.core.config` is
first imported anywhere, since Settings() is instantiated at import time.
That's why this happens at module level, at the very top of conftest.py,
which pytest always imports before collecting test modules.
"""
import os
import tempfile

_TMP_DATA_DIR = tempfile.mkdtemp(prefix="factlayer_test_")
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP_DATA_DIR}/test.db"
os.environ["DATA_DIR"] = _TMP_DATA_DIR
os.environ.setdefault("ANTHROPIC_API_KEY", "")  # tests must not hit the real API

import fitz  # noqa: E402
import pytest  # noqa: E402

from app.db.database import init_db  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _init_test_db():
    init_db()
    yield


def make_pdf(pages: list[str]) -> str:
    """Build a throwaway PDF with one page of plain text per string given."""
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=11)
    path = os.path.join(_TMP_DATA_DIR, f"test_{abs(hash(tuple(pages)))}.pdf")
    doc.save(path)
    doc.close()
    return path


@pytest.fixture()
def sample_pdf_path():
    return make_pdf(
        [
            "Revenue for the fiscal year ending December 2024 (FY2024) was $10 million.",
            "Total employee count at year end was 250 across all offices.",
        ]
    )

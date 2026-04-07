"""
_helpers.py — Shared test helper functions for the app test suite.

Provides common mock-building utilities used across multiple test modules to
avoid duplication.
"""

from unittest.mock import MagicMock


def make_fake_doc(
    content: str,
    source: str = "lasp_doc.pdf",
    page: int = 1,
    source_url: str = "",
) -> MagicMock:
    """Return a minimal LangChain Document-like mock.

    Shared by test_rag.py and test_discord_bot.py to avoid duplication.
    """
    doc = MagicMock()
    doc.page_content = content
    doc.metadata = {"source": source, "page": page, "source_url": source_url}
    return doc

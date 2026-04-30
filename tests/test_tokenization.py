"""
Tests for the Tokenization module.
"""

import sys
from typing import Any

import pytest

from gemma_4_sql.tokenization import SQLTokenizer


class MockHFTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [99, 100]

    def decode(self, tokens: list[int]) -> str:
        return "hf_decoded"


class MockAutoTokenizer:
    @classmethod
    def from_pretrained(cls, model_name: str) -> MockHFTokenizer:
        return MockHFTokenizer()


@pytest.fixture
def mock_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock the transformers library."""
    mock_transformers_module = type(
        "transformers", (), {"AutoTokenizer": MockAutoTokenizer}
    )
    monkeypatch.setitem(sys.modules, "transformers", mock_transformers_module)  # type: ignore[arg-type]

    import gemma_4_sql.tokenization

    monkeypatch.setattr(gemma_4_sql.tokenization, "AutoTokenizer", MockAutoTokenizer)


def test_sql_tokenizer_fallback() -> None:
    """Test fallback char-level encoding."""
    tok = SQLTokenizer()
    encoded = tok.encode("abc")
    assert encoded == [ord("a"), ord("b"), ord("c")]
    decoded = tok.decode(encoded)
    assert decoded == "abc"


def test_sql_tokenizer_hf(mock_transformers: Any) -> None:
    """Test Hugging Face tokenizer wrapping."""
    tok = SQLTokenizer(model_name="dummy/model")
    assert tok._hf_tokenizer is not None

    encoded = tok.encode("abc")
    assert encoded == [99, 100]

    decoded = tok.decode(encoded)
    assert decoded == "hf_decoded"

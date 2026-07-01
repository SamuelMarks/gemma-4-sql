"""Tests for the Tokenization module."""

import sys

import pytest

from gemma_4_sql.tokenization import SQLTokenizer


class MockHFTokenizer:
    """Initialize class MockHFTokenizer."""

    def encode(self, _text: str, **_kwargs: object) -> list[int]:
        """Initialize function encode.

        Args:
        ----
        text: Description of text.
        add_special_tokens: Description of add_special_tokens.

        """
        return [99, 100]

    def decode(self, _tokens: list[int]) -> str:
        """Initialize function decode.

        Args:
        ----
        tokens: Description of tokens.

        """
        return "hf_decoded"


class MockAutoTokenizer:
    """Initialize class MockAutoTokenizer."""

    @classmethod
    def from_pretrained(cls, _model_name: str) -> MockHFTokenizer:
        """Initialize function from_pretrained.

        Args:
        ----
        model_name: Description of model_name.

        """
        return MockHFTokenizer()


@pytest.fixture
def _mock_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock the transformers library."""
    mock_transformers_module = type("transformers", (), {"AutoTokenizer": MockAutoTokenizer})
    monkeypatch.setitem(sys.modules, "transformers", mock_transformers_module)
    gemma_4_sql = __import__("gemma_4_sql.tokenization")
    monkeypatch.setattr(gemma_4_sql.tokenization, "AutoTokenizer", MockAutoTokenizer)


def test_sql_tokenizer_fallback() -> None:
    """Test fallback char-level encoding."""
    tok = SQLTokenizer()
    encoded = tok.encode("abc")
    if not encoded == [ord("a"), ord("b"), ord("c")]:
        raise AssertionError
    decoded = tok.decode(encoded)
    if not decoded == "abc":
        raise AssertionError


@pytest.mark.usefixtures("_mock_transformers")
def test_sql_tokenizer_hf() -> None:
    """Test Hugging Face tokenizer wrapping."""
    tok = SQLTokenizer(model_name="dummy/model")
    if not tok.hf_tokenizer is not None:
        raise AssertionError
    encoded = tok.encode("abc")
    if not encoded == [99, 100]:
        raise AssertionError
    decoded = tok.decode(encoded)
    if not decoded == "hf_decoded":
        raise AssertionError

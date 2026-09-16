"""Tests for the Tokenization module."""

from __future__ import annotations

import sys

import pytest

from gemma_4_sql.tokenization import SQLTokenizer


class MockHFTokenizer:
    """Mock Hugging Face tokenizer implementation for testing."""

    def encode(self, _text: str, **_kwargs: object) -> list[int]:
        """Encode text using mock token IDs.

        Args:
            _text: Input text to encode.
            **_kwargs: Optional kwargs.

        Returns:
            List of mock token IDs.
        """
        return [99, 100]

    def decode(self, _tokens: list[int]) -> str:
        """Decode mock token IDs into text.

        Args:
            _tokens: Sequence of token IDs to decode.

        Returns:
            Decoded mock string.
        """
        return "hf_decoded"


class MockAutoTokenizer:
    """Mock AutoTokenizer class providing from_pretrained method."""

    @classmethod
    def from_pretrained(cls, _model_name: str) -> MockHFTokenizer:
        """Construct a MockHFTokenizer instance.

        Args:
            _model_name: Target model identifier.

        Returns:
            A new MockHFTokenizer instance.
        """
        return MockHFTokenizer()


class FailingAutoTokenizer:
    """Mock AutoTokenizer that raises an error on from_pretrained."""

    @classmethod
    def from_pretrained(cls, _model_name: str) -> MockHFTokenizer:
        """Raise an exception to test error handling.

        Args:
            _model_name: Target model identifier.

        Raises:
            OSError: Simulating model not found error.
        """
        msg = "Model not found"
        raise OSError(msg)


@pytest.fixture
def _mock_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock the transformers library with MockAutoTokenizer.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    mock_transformers_module = type("transformers", (), {"AutoTokenizer": MockAutoTokenizer})
    monkeypatch.setitem(sys.modules, "transformers", mock_transformers_module)
    gemma_4_sql = __import__("gemma_4_sql.tokenization")
    monkeypatch.setattr(gemma_4_sql.tokenization, "AutoTokenizer", MockAutoTokenizer)


def test_sql_tokenizer_fallback() -> None:
    """Test fallback byte/character-level encoding with ASCII and Unicode."""
    tok = SQLTokenizer()
    encoded = tok.encode("abc")
    assert encoded == [ord("a"), ord("b"), ord("c")]
    decoded = tok.decode(encoded)
    assert decoded == "abc"

    # Unicode & Emoji test
    unicode_text = "SELECT 'café' 🚀"
    unicode_encoded = tok.encode(unicode_text)
    assert len(unicode_encoded) > 0
    unicode_decoded = tok.decode(unicode_encoded)
    assert unicode_decoded == unicode_text


@pytest.mark.usefixtures("_mock_transformers")
def test_sql_tokenizer_hf() -> None:
    """Test Hugging Face tokenizer wrapping."""
    tok = SQLTokenizer(model_name="dummy/model")
    assert tok.hf_tokenizer is not None
    encoded = tok.encode("abc")
    assert encoded == [99, 100]
    decoded = tok.decode(encoded)
    assert decoded == "hf_decoded"


def test_sql_tokenizer_hf_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test tokenizer initialization when from_pretrained raises an exception.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    gemma_4_sql = __import__("gemma_4_sql.tokenization")
    monkeypatch.setattr(gemma_4_sql.tokenization, "AutoTokenizer", FailingAutoTokenizer)
    tok = SQLTokenizer(model_name="nonexistent/model")
    assert tok.hf_tokenizer is None
    # Verifies fallback works when HF tokenizer fails to load
    assert tok.encode("abc") == [ord("a"), ord("b"), ord("c")]


def test_sql_tokenizer_fallback_edge_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test edge cases in fallback encoding and decoding.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    tok = SQLTokenizer(vocab_size=128)

    # Test when string encode raises UnicodeEncodeError
    class BadString:
        """Object that fails utf-8 encode."""

        def encode(self, _encoding: str) -> bytes:
            """Raise UnicodeEncodeError.

            Args:
                _encoding: Encoding name.

            Raises:
                UnicodeEncodeError: Simulated encoding error.
            """
            raise UnicodeEncodeError("utf-8", "", 0, 1, "test")

        def __str__(self) -> str:
            """String representation.

            Returns:
                Test string.
            """
            return "abc"

    encoded = tok.encode(BadString())  # type: ignore[arg-type]
    assert encoded == [ord("a"), ord("b"), ord("c")]

    # Test when both encode and str(text).encode fail
    class ReallyBadString:
        """Object that fails both direct encode and str encode."""

        def encode(self, _encoding: str, **kwargs: object) -> bytes:
            raise UnicodeEncodeError("utf-8", "", 0, 1, "test")

        def __str__(self) -> str:
            class FailingStr(str):
                def encode(self, _encoding: str, **kwargs: object) -> bytes:
                    raise RuntimeError("Str encode failed")

            return FailingStr("abc")

    encoded_really_bad = tok.encode(ReallyBadString())  # type: ignore[arg-type]
    assert encoded_really_bad == [ord("a") % 128, ord("b") % 128, ord("c") % 128]

    # Test decode when bytes(...) raises ValueError/TypeError
    bad_tokens = [object()]  # type: ignore[list-item]
    decoded_bad = tok.decode(bad_tokens)  # type: ignore[arg-type]
    assert decoded_bad == ""


def test_sql_tokenizer_get_vocab() -> None:
    """Test get_vocab for fallback byte-level and mock HF tokenizer."""
    tok = SQLTokenizer(vocab_size=128)
    vocab = tok.get_vocab()
    assert isinstance(vocab, dict)
    assert len(vocab) == 128
    assert vocab["a"] == ord("a")

    class MockVocabHFTokenizer(MockHFTokenizer):
        def get_vocab(self) -> dict[str, int]:
            return {"<pad>": 0, "select": 1}

    tok_hf = SQLTokenizer()
    tok_hf.hf_tokenizer = MockVocabHFTokenizer()
    hf_vocab = tok_hf.get_vocab()
    assert hf_vocab == {"<pad>": 0, "select": 1}


def test_sql_tokenizer_lazy_getattr(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test tokenizer init when AutoTokenizer is None and resolved via getattr.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    import gemma_4_sql.tokenization as tok_mod

    monkeypatch.setattr(tok_mod, "AutoTokenizer", None)
    monkeypatch.setattr(tok_mod._transformers_mod, "AutoTokenizer", MockAutoTokenizer, raising=False)
    tok = tok_mod.SQLTokenizer(model_name="dummy")
    assert tok.hf_tokenizer is not None


def test_sql_tokenizer_empty_and_special_sql_chars() -> None:
    """Test SQLTokenizer with empty list, special SQL characters, and long text."""
    tok = SQLTokenizer()
    # Empty list
    assert tok.decode([]) == ""

    # Special SQL characters
    special_sql = "SELECT * FROM \"my_table\" WHERE name = 'O\\'Reilly' AND x >= 10; -- comment\n/* block */"
    encoded = tok.encode(special_sql)
    assert len(encoded) > 0
    decoded = tok.decode(encoded)
    assert decoded == special_sql

    # Long text exceeding vocabulary size / boundaries
    long_query = "SELECT " + ", ".join(f"col_{i}" for i in range(500)) + " FROM very_large_table;"
    enc_long = tok.encode(long_query)
    assert len(enc_long) > 0
    assert tok.decode(enc_long) == long_query

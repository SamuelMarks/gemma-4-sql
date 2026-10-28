"""Tests for the Tokenization module."""

from __future__ import annotations

import math
import sys

import pytest

from gemma_4_sql.exceptions import ConfigurationError, DependencyMissingError
from gemma_4_sql.tokenization import ByteFallbackTokenizer, SQLTokenizer


class MockHFTokenizer:
    """Mock Hugging Face tokenizer implementation for testing."""

    pad_token_id: int | None = 0
    eos_token_id: int | None = 2
    bos_token_id: int | None = 1

    def encode(self, text: str, **kwargs: bool | int | str | None) -> list[int]:
        """Encode text using mock token IDs.

        Args:
            text: Input text to encode.
            **kwargs: Optional kwargs.

        Returns:
            List of mock token IDs.

        """
        _ = (text, kwargs)
        return [99, 100]

    def decode(self, token_ids: list[int]) -> str:
        """Decode mock token IDs into text.

        Args:
            token_ids: Sequence of token IDs to decode.

        Returns:
            Decoded mock string.

        """
        return f"hf_decoded:{token_ids}"

    def get_vocab(self) -> dict[str, int]:
        """Return mock vocabulary mapping."""
        return {"<pad>": 0, "select": 1}


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
    import gemma_4_sql.tokenization as tok_mod

    mock_transformers_module = type("transformers", (), {"AutoTokenizer": MockAutoTokenizer})
    monkeypatch.setitem(sys.modules, "transformers", mock_transformers_module)
    monkeypatch.setattr(tok_mod, "AutoTokenizer", MockAutoTokenizer)


def test_sql_tokenizer_fallback() -> None:
    """Test fallback byte/character-level encoding with ASCII, Unicode, and token ID properties."""
    tok = SQLTokenizer(pad_token_id=3, eos_token_id=4, bos_token_id=5)
    assert tok.pad_token_id == 3
    assert tok.eos_token_id == 4
    assert tok.bos_token_id == 5
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

    # Batch sequence encoding test
    batch_encoded = tok.encode(["ab", "cd"])
    assert batch_encoded == [[ord("a"), ord("b")], [ord("c"), ord("d")]]


@pytest.mark.usefixtures("_mock_transformers")
def test_sql_tokenizer_hf() -> None:
    """Test Hugging Face tokenizer wrapping, special token properties, and float rejection."""
    tok = SQLTokenizer(model_name="dummy/model")
    assert tok.hf_tokenizer is not None
    assert tok.pad_token_id == 0
    assert tok.eos_token_id == 2
    assert tok.bos_token_id == 1
    assert tok.get_vocab() == {"<pad>": 0, "select": 1}
    encoded = tok.encode("abc")
    assert encoded == [99, 100]
    decoded = tok.decode([99, 1.5, "100", "bad"])
    assert decoded == "hf_decoded:[99, 100]"

    # Test when HF tokenizer special token IDs are None (falls back to byte_fallback properties)
    hf_tok = MockHFTokenizer()
    hf_tok.pad_token_id = None
    hf_tok.eos_token_id = None
    hf_tok.bos_token_id = None
    tok.hf_tokenizer = hf_tok
    assert tok.pad_token_id == 0
    assert tok.eos_token_id == 1
    assert tok.bos_token_id == 2


def test_sql_tokenizer_hf_failure(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """Test tokenizer initialization when from_pretrained raises an exception and logs a warning.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        caplog: Pytest log capture fixture.

    """
    import gemma_4_sql.tokenization as tok_mod

    monkeypatch.setattr(tok_mod, "AutoTokenizer", FailingAutoTokenizer)
    tok = SQLTokenizer(model_name="nonexistent/model")
    assert tok.hf_tokenizer is None
    assert "Falling back to deterministic byte-level SQLTokenizer" in caplog.text
    assert tok.encode("abc") == [ord("a"), ord("b"), ord("c")]


def test_sql_tokenizer_get_vocab() -> None:
    """Test get_vocab for fallback byte-level and mock HF tokenizer."""
    tok = SQLTokenizer(vocab_size=128)
    vocab = tok.get_vocab()
    assert isinstance(vocab, dict)
    assert len(vocab) == 128
    assert vocab["a"] == ord("a")

    tok_hf = SQLTokenizer()
    tok_hf.hf_tokenizer = MockHFTokenizer()
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
    assert tok.decode([]) == ""

    special_sql = """SELECT * FROM "my_table" WHERE name = 'O''Reilly' AND x >= 10; -- comment\n/* block */"""
    encoded = tok.encode(special_sql)
    assert len(encoded) > 0
    decoded = tok.decode(encoded)
    assert decoded == special_sql

    long_query = "SELECT " + ", ".join(f"col_{i}" for i in range(500)) + " FROM very_large_table;"
    enc_long = tok.encode(long_query)
    assert len(enc_long) > 0
    assert tok.decode(enc_long) == long_query


def test_sql_tokenizer_unicode_multibyte() -> None:
    """Test SQLTokenizer handling of multi-byte UTF-8, emojis, and non-ASCII SQL comments."""
    tok = SQLTokenizer()
    text = "SELECT * FROM users WHERE name = '田中' AND bio = '🚀 Data Scientist! 🎉'; -- 数据库查询"
    encoded = tok.encode(text)
    assert len(encoded) > len(text)
    decoded = tok.decode(encoded)
    assert decoded == text


def test_sql_tokenizer_custom_vocab_bounds() -> None:
    """Test SQLTokenizer with small custom vocab_size, reversible nibble expansion, and vocab_size=1."""
    tok_small = SQLTokenizer(vocab_size=64)
    assert tok_small.vocab_size == 64
    encoded = tok_small.encode("ABCabc123")
    for tid in encoded:
        assert 0 <= tid < 64
    assert tok_small.decode(encoded) == "ABCabc123"

    decoded_exact = tok_small.decode([1, 1, 1, 2])
    assert isinstance(decoded_exact, str)
    out_of_bound_ids = [-5, 0, 65, 255, 300]
    decoded = tok_small.decode(out_of_bound_ids)
    assert isinstance(decoded, str)

    tok_one = SQLTokenizer(vocab_size=1)
    assert tok_one.encode("hi") == [0, 0]
    assert tok_one.decode([0, 0]) == "\x00\x00"


def test_tokenizer_decode_ignores_non_integer_and_float_tokens() -> None:
    """Verify SQLTokenizer.decode ignores float and non-integer elements in token sequence."""
    tokenizer = SQLTokenizer(vocab_size=256)
    mixed_tokens: list[int | str | float] = [65, math.pi, "bad", 66]
    res = tokenizer.decode(mixed_tokens)
    assert res == "AB"

    bf = ByteFallbackTokenizer(vocab_size=256, pad_token_id=7, eos_token_id=8, bos_token_id=9)
    assert bf.pad_token_id == 7
    assert bf.eos_token_id == 8
    assert bf.bos_token_id == 9

    with pytest.raises(ConfigurationError, match="vocab_size must be positive"):
        SQLTokenizer(vocab_size=0, strict_vocab=True)

    with pytest.raises(DependencyMissingError, match="allow_fallback=False"):
        SQLTokenizer(model_name="nonexistent/model/path/for/test", allow_fallback=False)

    with pytest.raises(DependencyMissingError, match="allow_fallback=False"):
        SQLTokenizer(model_name="nonexistent/model/path/for/test", allow_byte_fallback=False)

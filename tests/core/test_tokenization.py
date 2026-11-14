"""Unit and behavioral tests for the Tokenization module."""

from __future__ import annotations

import math

import pytest

import gemma_4_sql.tokenization as tok_mod
from gemma_4_sql.backends.lazy_loader import override_lazy_module
from gemma_4_sql.exceptions import ConfigurationError, DependencyMissingError
from gemma_4_sql.tokenization import (
    DEFAULT_BYTE_VOCAB_SIZE,
    ByteFallbackTokenizer,
    SQLTokenizer,
    get_auto_tokenizer_class,
)


class DummyHFTokenizer:
    """Typed stub mimicking a Hugging Face PreTrainedTokenizer."""

    pad_token_id: int | None = 10
    eos_token_id: int | None = 11
    bos_token_id: int | None = 12

    def __init__(self, pretrained_model_name_or_path: str = "dummy") -> None:
        """Initialize stub tokenizer.

        Args:
            pretrained_model_name_or_path: Model identifier string.

        """
        self.name = pretrained_model_name_or_path

    def get_vocab(self) -> dict[str, int]:
        """Return dummy vocabulary.

        Returns:
            Dummy vocabulary dictionary.

        """
        return {"SELECT": 1, "*": 2, "FROM": 3}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode string by splitting on spaces into word lengths.

        Args:
            text: Input text string.
            add_special_tokens: Unused flag matching HF API.

        Returns:
            List of word lengths.

        """
        _ = add_special_tokens
        return [len(word) for word in text.split()]

    def decode(self, tokens: list[int]) -> str:
        """Decode integer tokens into space-separated strings.

        Args:
            tokens: Token ID sequence.

        Returns:
            Space-joined token string.

        """
        return " ".join(str(t) for t in tokens)


class DummyAutoTokenizer:
    """Typed stub mimicking `transformers.AutoTokenizer`."""

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str) -> DummyHFTokenizer:
        """Load dummy HF tokenizer or raise OSError when requested.

        Args:
            pretrained_model_name_or_path: Model identifier string.

        Returns:
            Loaded `DummyHFTokenizer` instance.

        Raises:
            OSError: When `pretrained_model_name_or_path == 'raise_error'`.

        """
        if pretrained_model_name_or_path == "raise_error":
            raise OSError("Model not found")
        return DummyHFTokenizer(pretrained_model_name_or_path)


def test_tokenizer_byte_fallback_disjoint_special_tokens() -> None:
    """Verify ByteFallbackTokenizer offsets raw UTF-8 bytes away from PAD/EOS/BOS control IDs."""
    tok = SQLTokenizer(pad_token_id=0, eos_token_id=1, bos_token_id=2)
    assert tok.vocab_size == DEFAULT_BYTE_VOCAB_SIZE == 259
    assert tok.hf_tokenizer is None
    assert tok.is_fallback is True
    assert tok.is_byte_fallback is True
    assert tok.pad_token_id == 0
    assert tok.eos_token_id == 1
    assert tok.bos_token_id == 2
    assert tok.byte_fallback.byte_offset == 3

    control_payload = "\x00\x01\x02SELECT 1;"
    encoded_ctrl = tok.encode(control_payload)
    assert all(tid not in {0, 1, 2} for tid in encoded_ctrl)
    assert encoded_ctrl[:3] == [3, 4, 5]
    assert tok.decode(encoded_ctrl) == control_payload

    framed = [tok.bos_token_id, *encoded_ctrl, tok.eos_token_id, tok.pad_token_id]
    assert tok.decode(framed) == control_payload

    tok_custom_specials = SQLTokenizer(pad_token_id=3, eos_token_id=4, bos_token_id=5)
    assert tok_custom_specials.pad_token_id == 3
    assert tok_custom_specials.eos_token_id == 4
    assert tok_custom_specials.bos_token_id == 5

    vocab = tok_custom_specials.get_vocab()
    assert len(vocab) == DEFAULT_BYTE_VOCAB_SIZE
    assert vocab["<pad>"] == 3
    assert vocab["<eos>"] == 4
    assert vocab["<bos>"] == 5
    assert vocab["a"] == ord("a") + 3

    encoded = tok_custom_specials.encode("abc")
    assert all(tid not in {3, 4, 5} for tid in encoded)
    assert tok_custom_specials.decode(encoded) == "abc"

    batch_encoded = tok_custom_specials.encode(["ab", "c"])
    assert len(batch_encoded) == 2
    assert tok_custom_specials.decode(batch_encoded[0]) == "ab"
    assert tok_custom_specials.decode(batch_encoded[1]) == "c"


def test_byte_to_token_ids_variants() -> None:
    """Verify `byte_to_token_ids` handles direct 1-token, multi-token radix, and 1-token degenerate vocabs."""
    bf_default = ByteFallbackTokenizer()
    assert bf_default.byte_to_token_ids(0) == [3]
    assert bf_default.byte_to_token_ids(65) == [68]

    bf_radix = ByteFallbackTokenizer(vocab_size=64)
    ids_65 = bf_radix.byte_to_token_ids(65)
    assert len(ids_65) == 2
    assert all(tid not in {0, 1, 2} and 0 <= tid < 64 for tid in ids_65)
    assert bf_radix.decode_tokens(ids_65) == "A"

    bf_one = ByteFallbackTokenizer(vocab_size=1)
    assert bf_one.byte_to_token_ids(65) == [0]


def test_tokenizer_hf_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify SQLTokenizer delegates to Hugging Face AutoTokenizer when available."""
    monkeypatch.setattr(tok_mod, "AutoTokenizer", DummyAutoTokenizer)

    tok = SQLTokenizer(model_name="my-hf-model")
    assert tok.hf_tokenizer is not None
    assert tok.is_fallback is False
    assert tok.pad_token_id == 10
    assert tok.eos_token_id == 11
    assert tok.bos_token_id == 12
    assert tok.get_vocab() == {"SELECT": 1, "*": 2, "FROM": 3}
    assert tok.encode("SELECT * FROM table") == [6, 1, 4, 5]
    assert tok.encode(["SELECT *", "FROM"]) == [[6, 1], [4]]
    assert tok.decode([6, math.e, "bad", 1, 4]) == "6 1 4"


def test_tokenizer_hf_missing_special_attributes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify SQLTokenizer falls back to byte_fallback special IDs when HF tokenizer IDs are None."""

    class NoneIdsHFTokenizer:
        """HF tokenizer stub whose special ID properties are None."""

        pad_token_id: int | None = None
        eos_token_id: int | None = None
        bos_token_id: int | None = None

        def encode(self, text: str, **_kwargs: bool | int | str | None) -> list[int]:
            """Encode text length.

            Returns:
                Single-element list with input text length.

            """
            return [len(text)]

        def decode(self, token_ids: list[int]) -> str:
            """Decode token IDs.

            Returns:
                Comma-joined string of token IDs.

            """
            return ",".join(str(x) for x in token_ids)

        def get_vocab(self) -> dict[str, int]:
            """Return minimal vocab."""
            return {"a": 1}

    class NonIntIdsAutoTokenizer:
        """AutoTokenizer stub returning `NoneIdsHFTokenizer`."""

        @classmethod
        def from_pretrained(cls, _pretrained_model_name_or_path: str) -> NoneIdsHFTokenizer:
            """Return `NoneIdsHFTokenizer`."""
            return NoneIdsHFTokenizer()

    monkeypatch.setattr(tok_mod, "AutoTokenizer", NonIntIdsAutoTokenizer)
    tok = SQLTokenizer(vocab_size=128, model_name="none-ids-model")
    assert tok.pad_token_id == 0
    assert tok.eos_token_id == 1
    assert tok.bos_token_id == 2
    assert tok.get_vocab() == {"a": 1}
    assert tok.encode("abc") == [3]
    assert tok.decode([3]) == "3"

    tok_fallback = SQLTokenizer(vocab_size=128)
    assert tok_fallback.pad_token_id == 0
    assert tok_fallback.eos_token_id == 1
    assert tok_fallback.bos_token_id == 2
    assert len(tok_fallback.get_vocab()) == 128


def test_tokenizer_hf_fallback_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify SQLTokenizer falls back to ByteFallbackTokenizer when AutoTokenizer.from_pretrained raises OSError."""
    monkeypatch.setattr(tok_mod, "AutoTokenizer", DummyAutoTokenizer)

    tok = SQLTokenizer(model_name="raise_error")
    assert tok.hf_tokenizer is None
    assert tok.is_fallback is True
    encoded = tok.encode("ab")
    assert tok.decode(encoded) == "ab"


def test_lazy_get_auto_tokenizer_class(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `get_auto_tokenizer_class` resolves `transformers.AutoTokenizer` lazily without import-time loading."""
    from tests.utils.test_utils import FakeModuleNamespace

    class _TypedTransformersModule(FakeModuleNamespace):
        """Typed module double exposing `AutoTokenizer`."""

        AutoTokenizer: type[DummyAutoTokenizer] = DummyAutoTokenizer

    fake_tf_mod = _TypedTransformersModule("transformers")
    empty_mod = FakeModuleNamespace("transformers")

    monkeypatch.setattr(tok_mod, "AutoTokenizer", None)
    with override_lazy_module("transformers", fake_tf_mod):
        assert get_auto_tokenizer_class() is DummyAutoTokenizer

    with override_lazy_module("transformers", empty_mod):
        assert get_auto_tokenizer_class() is None
        tok = SQLTokenizer(model_name="some-model")
        assert tok.hf_tokenizer is None


def test_tokenizer_utf8_and_vocab_bounds() -> None:
    """Verify multi-byte UTF-8 roundtrips and out-of-bounds/partial radix chunks decode safely."""
    tok = SQLTokenizer()
    sql = "SELECT '🚀 café' FROM users WHERE name = 'Müller';"
    tokens = tok.encode(sql)
    assert all(0 <= tid < tok.vocab_size and tid not in {0, 1, 2} for tid in tokens)
    assert tok.decode(tokens) == sql

    assert isinstance(tok.decode([-5, 300, 999]), str)

    tok_small = SQLTokenizer(vocab_size=64)
    encoded = tok_small.encode("ABCabc123")
    assert all(0 <= tid < 64 and tid not in {0, 1, 2} for tid in encoded)
    assert tok_small.decode(encoded) == "ABCabc123"

    # Partial trailing chunk (< tokens_per_byte) branch
    assert isinstance(tok_small.decode([4, 5, 6]), str)
    assert isinstance(tok_small.decode([-5, 0, 65, 255, 300]), str)

    tok_one = SQLTokenizer(vocab_size=1)
    assert tok_one.encode("hi") == [0, 0]
    assert tok_one.decode([0, 0]) == "\x00\x00"


def test_tokenizer_decode_ignores_non_integer_and_float_tokens() -> None:
    """Verify SQLTokenizer.decode ignores float and non-integer elements in token sequence."""
    tokenizer = SQLTokenizer()
    a_id = tokenizer.byte_fallback.byte_to_token_ids(ord("A"))[0]
    b_id = tokenizer.byte_fallback.byte_to_token_ids(ord("B"))[0]
    mixed_tokens: list[int | str | float] = [a_id, math.pi, "bad", b_id]
    assert tokenizer.decode(mixed_tokens) == "AB"

    bf = ByteFallbackTokenizer(vocab_size=259, pad_token_id=7, eos_token_id=7, bos_token_id=9)
    assert bf.pad_token_id == 7
    assert bf.eos_token_id == 7
    assert bf.bos_token_id == 9
    assert len(bf.get_vocab()) == 259

    with pytest.raises(ConfigurationError, match="vocab_size must be positive"):
        SQLTokenizer(vocab_size=0, strict_vocab=True)

    with pytest.raises(DependencyMissingError, match="allow_fallback=False"):
        SQLTokenizer(model_name="nonexistent/model/path/for/test", allow_fallback=False)

    with pytest.raises(DependencyMissingError, match="allow_fallback=False"):
        SQLTokenizer(model_name="nonexistent/model/path/for/test", allow_byte_fallback=False)


@pytest.mark.parametrize("vocab_size", [2, 3, 7, 16, 128, 255, 256, 259, 512])
def test_byte_fallback_radix_roundtrip_across_vocab_sizes(vocab_size: int) -> None:
    """Verify `ByteFallbackTokenizer` reversibly encodes and decodes ASCII and UTF-8 across arbitrary radix bases."""
    bf = ByteFallbackTokenizer(vocab_size=vocab_size)
    sample = "SELECT id, 'Ω≈ç√' FROM t WHERE x = 42;"
    tokens = bf.encode_str(sample)
    assert all(0 <= t < vocab_size for t in tokens)
    if vocab_size >= 5:
        assert all(t not in {bf.pad_token_id, bf.eos_token_id, bf.bos_token_id} for t in tokens)
    assert bf.decode_tokens(tokens) == sample
    vocab = bf.get_vocab()
    assert len(vocab) == vocab_size

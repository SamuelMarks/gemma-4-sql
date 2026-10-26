"""Tokenization module for processing Text-to-SQL datasets."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, SupportsInt, overload

from gemma_4_sql.backends.lazy_loader import LazyLoader
from gemma_4_sql.exceptions import ConfigurationError, DependencyMissingError
from gemma_4_sql.type_hints import HFTokenizerProtocol

logger = logging.getLogger(__name__)
_transformers_mod: Any = LazyLoader("transformers").get_module()
AutoTokenizer: Any = getattr(_transformers_mod, "AutoTokenizer", None) if _transformers_mod is not None else None


class ByteFallbackTokenizer:
    """Deterministic, reversible UTF-8 byte-level fallback tokenizer implementing `TokenizerProtocol`.

    When `vocab_size >= 256`, each UTF-8 byte maps 1:1 to its integer byte value in `[0, 255]`.
    When `2 <= vocab_size < 256`, each UTF-8 byte is reversibly decomposed into fixed-width
    base-`vocab_size` digits so all token IDs remain strictly within `[0, vocab_size)` without
    destructive hash collisions.
    """

    def __init__(self, vocab_size: int = 256) -> None:
        """Initialize the byte-level fallback tokenizer.

        Args:
            vocab_size: Maximum token ID bound (`>= 1`).

        """
        self.vocab_size = max(1, int(vocab_size))
        self.tokens_per_byte = 1
        if 2 <= self.vocab_size < 256:
            count = 1
            cap = self.vocab_size
            while cap < 256:
                cap *= self.vocab_size
                count += 1
            self.tokens_per_byte = count

    def get_vocab(self) -> dict[str, int]:
        """Return the vocabulary mapping tokens or byte labels to integer IDs.

        Returns:
            Dictionary mapping printable ASCII characters or `<byte_N>` tags to IDs in `[0, vocab_size)`.

        """
        vocab: dict[str, int] = {}
        for b in range(self.vocab_size):
            vocab[chr(b) if 32 <= b <= 126 else f"<byte_{b}>"] = b
        return vocab

    def encode_str(self, text: str) -> list[int]:
        """Encode a single UTF-8 string into a list of integer token IDs in `[0, vocab_size)`.

        Args:
            text: Input string to tokenize.

        Returns:
            List of integer token IDs.

        """
        raw_bytes = str(text).encode("utf-8", errors="replace")
        if self.vocab_size >= 256:
            return list(raw_bytes)
        if self.vocab_size == 1:
            return [0 for _ in raw_bytes]

        tokens: list[int] = []
        base = self.vocab_size
        k = self.tokens_per_byte
        for b in raw_bytes:
            val = b
            digits = [0] * k
            for pos in range(k - 1, -1, -1):
                digits[pos] = val % base
                val //= base
            tokens.extend(digits)
        return tokens

    def decode_tokens(self, tokens: Sequence[int | object]) -> str:
        """Decode a sequence of integer token IDs back into a UTF-8 string.

        Args:
            tokens: Sequence of integer token IDs.

        Returns:
            Decoded UTF-8 string.

        """
        parsed_ints: list[int] = []
        for t in tokens:
            try:
                t_int = int(str(t)) if not isinstance(t, (int, bytes, SupportsInt)) else int(t)
            except (ValueError, TypeError):
                continue
            parsed_ints.append(t_int)

        if self.vocab_size >= 256 or self.vocab_size == 1:
            valid_bytes = [t % 256 for t in parsed_ints]
            return bytes(valid_bytes).decode("utf-8", errors="replace")

        base = self.vocab_size
        k = self.tokens_per_byte
        reconstructed: list[int] = []
        for idx in range(0, len(parsed_ints), k):
            chunk = parsed_ints[idx : idx + k]
            if len(chunk) < k:
                reconstructed.extend(val % 256 for val in chunk)
                break
            byte_val = 0
            for digit in chunk:
                byte_val = byte_val * base + (digit % base)
            reconstructed.append(byte_val % 256)

        return bytes(reconstructed).decode("utf-8", errors="replace")


class SQLTokenizer:
    """A tokenizer for SQL datasets.

    Wraps a Hugging Face tokenizer (like SentencePiece for Gemma) if available
    and a `model_name` is provided. Otherwise, delegates to `ByteFallbackTokenizer`
    for a deterministic, reversible UTF-8 byte-level encoding scheme.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        model_name: str | None = None,
        *,
        strict_vocab: bool = False,
        allow_fallback: bool = True,
    ) -> None:
        """Initialize the tokenizer.

        Args:
            vocab_size: Vocabulary size for the byte-level fallback tokenizer.
            model_name: Optional model identifier or path to a Hugging Face tokenizer.
            strict_vocab: If True, raise `ConfigurationError` when `vocab_size <= 0`.
            allow_fallback: If False and `model_name` is set, raise `DependencyMissingError` when HF tokenizer fails to load.

        Raises:
            ConfigurationError: If `strict_vocab=True` and `vocab_size <= 0`.
            DependencyMissingError: If `allow_fallback=False` and the Hugging Face tokenizer cannot be loaded.

        """
        if strict_vocab and int(vocab_size) <= 0:
            raise ConfigurationError(f"vocab_size must be positive, got {vocab_size}")
        self.vocab_size = max(1, int(vocab_size))
        self.model_name = model_name
        self.allow_fallback = allow_fallback
        self.byte_fallback = ByteFallbackTokenizer(vocab_size=self.vocab_size)
        self.hf_tokenizer: HFTokenizerProtocol | None = self._try_load_hf_tokenizer()
        self.is_fallback: bool = self.hf_tokenizer is None
        if self.model_name and self.hf_tokenizer is None:
            if not self.allow_fallback:
                raise DependencyMissingError(
                    f"Failed to load Hugging Face tokenizer for '{self.model_name}' and allow_fallback=False.",
                    dependency_name="transformers",
                )
            logger.warning("Falling back to deterministic byte-level SQLTokenizer for '%s'.", self.model_name)

    def _try_load_hf_tokenizer(self) -> HFTokenizerProtocol | None:
        """Attempt to load a Hugging Face `AutoTokenizer` if `model_name` is set.

        Returns:
            Loaded Hugging Face tokenizer instance or `None` if unavailable.

        """
        tok_cls = AutoTokenizer
        if tok_cls is None and _transformers_mod is not None:
            tok_cls = getattr(_transformers_mod, "AutoTokenizer", None)
        if not self.model_name or tok_cls is None or not hasattr(tok_cls, "from_pretrained"):
            return None
        try:
            loaded: HFTokenizerProtocol = tok_cls.from_pretrained(self.model_name)
            return loaded
        except (ImportError, OSError, ValueError, TypeError, RuntimeError, AttributeError):
            return None

    def get_vocab(self) -> dict[str, int]:
        """Return the vocabulary mapping tokens or bytes to token IDs.

        Returns:
            A dictionary mapping token strings or byte representations to integer IDs.

        """
        if self.hf_tokenizer is not None and hasattr(self.hf_tokenizer, "get_vocab"):
            return dict(self.hf_tokenizer.get_vocab())
        return self.byte_fallback.get_vocab()

    @overload
    def encode(self, text: str) -> list[int]:
        """Encode a single string into a list of integer token IDs."""

    @overload
    def encode(self, text: list[str] | tuple[str, ...]) -> list[list[int]]:
        """Encode a sequence of strings into a list of integer token ID lists."""

    def encode(self, text: str | Sequence[str]) -> list[int] | list[list[int]]:
        """Encode a string or sequence of strings into token IDs.

        Args:
            text: Single string or sequence of strings to be tokenized.

        Returns:
            A list of integer token IDs for a single string, or a list of token ID lists for a sequence of strings.

        """
        if not isinstance(text, str):
            return [self.encode(item) for item in text]

        if self.hf_tokenizer is not None:
            encoded: Sequence[int] = self.hf_tokenizer.encode(str(text), add_special_tokens=False)
            return [int(x) for x in encoded]

        return self.byte_fallback.encode_str(text)

    def decode(self, tokens: Sequence[int | object]) -> str:
        """Decode a sequence of token IDs back into a UTF-8 string.

        Args:
            tokens: A sequence of integer token IDs.

        Returns:
            The decoded text string.

        """
        if self.hf_tokenizer is not None:
            int_tokens = [int(t) for t in tokens if isinstance(t, (int, float, str))]
            return str(self.hf_tokenizer.decode(int_tokens))

        return self.byte_fallback.decode_tokens(tokens)


__all__ = ["ByteFallbackTokenizer", "SQLTokenizer"]

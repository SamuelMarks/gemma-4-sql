"""Tokenization module for processing Text-to-SQL datasets."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, SupportsInt

from gemma_4_sql.backends.lazy_loader import LazyLoader

_transformers_mod: Any = LazyLoader("transformers").get_module()
AutoTokenizer: Any = getattr(_transformers_mod, "AutoTokenizer", None) if _transformers_mod is not None else None


class SQLTokenizer:
    """A tokenizer for SQL datasets.

    Wraps a Hugging Face tokenizer (like SentencePiece for Gemma) if available
    and a `model_name` is provided. Otherwise, falls back to a deterministic,
    reversible UTF-8 byte-level encoding scheme.
    """

    def __init__(self, vocab_size: int = 256, model_name: str | None = None) -> None:
        """Initialize the tokenizer.

        Args:
            vocab_size: Vocabulary size for the byte-level fallback tokenizer.
            model_name: Optional model identifier or path to a Hugging Face tokenizer.

        """
        self.vocab_size = max(1, int(vocab_size))
        self.model_name = model_name
        self.hf_tokenizer: Any = self._try_load_hf_tokenizer()

    def _try_load_hf_tokenizer(self) -> Any:
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
            return tok_cls.from_pretrained(self.model_name)
        except (ImportError, OSError, ValueError, TypeError, RuntimeError, AttributeError):
            return None

    def get_vocab(self) -> dict[str, int]:
        """Return the vocabulary mapping tokens or bytes to token IDs.

        Returns:
            A dictionary mapping token strings or byte representations to integer IDs.

        """
        if self.hf_tokenizer is not None and hasattr(self.hf_tokenizer, "get_vocab"):
            return dict(self.hf_tokenizer.get_vocab())

        vocab: dict[str, int] = {}
        for b in range(self.vocab_size):
            vocab[chr(b) if 32 <= b <= 126 else f"<byte_{b}>"] = b
        return vocab

    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of token IDs.

        Args:
            text: The string to be tokenized.

        Returns:
            A list of integer token IDs.

        """
        if self.hf_tokenizer is not None:
            encoded: Sequence[int] = self.hf_tokenizer.encode(text, add_special_tokens=False)
            return [int(x) for x in encoded]

        return [b % self.vocab_size for b in str(text).encode("utf-8", errors="replace")]

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

        modulus = min(256, self.vocab_size)
        valid_bytes: list[int] = []
        for t in tokens:
            try:
                t_int = int(str(t)) if not isinstance(t, (int, bytes, SupportsInt)) else int(t)
            except (ValueError, TypeError):
                continue
            valid_bytes.append(t_int % modulus)

        return bytes(valid_bytes).decode("utf-8", errors="replace")

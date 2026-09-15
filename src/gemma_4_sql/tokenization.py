"""Tokenization module for processing Text-to-SQL datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from gemma_4_sql.backends.lazy_loader import LazyLoader

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

_transformers_mod = LazyLoader("transformers").get_module()
AutoTokenizer: Any = getattr(_transformers_mod, "AutoTokenizer", None) if _transformers_mod is not None else None


class SQLTokenizer:
    """A tokenizer for SQL datasets.

    Wraps a Hugging Face tokenizer (like SentencePiece for Gemma) if available
    and a model_name is provided. Otherwise, falls back to a byte/character-level
    encoding scheme.
    """

    def __init__(self, vocab_size: int = 256, model_name: str | None = None) -> None:
        """Initialize the tokenizer.

        Args:
            vocab_size: The integer value for vocabulary size.
            model_name: Optional model identifier or path to Hugging Face tokenizer.
        """
        self.vocab_size = vocab_size
        self.model_name = model_name
        self.hf_tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None

        tok_cls = AutoTokenizer
        if tok_cls is None and _transformers_mod is not None:
            tok_cls = getattr(_transformers_mod, "AutoTokenizer", None)

        if self.model_name and tok_cls is not None and hasattr(tok_cls, "from_pretrained"):
            try:
                self.hf_tokenizer = tok_cls.from_pretrained(self.model_name)
            except (ImportError, OSError, ValueError, TypeError, RuntimeError, AttributeError):
                self.hf_tokenizer = None

    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of token IDs.

        Args:
            text: The string to be tokenized.

        Returns:
            A list of integer token IDs.
        """
        if self.hf_tokenizer is not None:
            return cast("list[int]", self.hf_tokenizer.encode(text, add_special_tokens=False))
        try:
            return list(text.encode("utf-8"))
        except (UnicodeEncodeError, AttributeError):
            return [ord(c) % self.vocab_size for c in str(text)]

    def decode(self, tokens: list[int]) -> str:
        """Decode a list of token IDs back into a string.

        Args:
            tokens: A sequence of integer token IDs.

        Returns:
            The decoded text string.
        """
        if self.hf_tokenizer is not None:
            return cast("str", self.hf_tokenizer.decode(tokens))
        try:
            return bytes([int(t) % 256 for t in tokens]).decode("utf-8", errors="replace")
        except (ValueError, TypeError):
            return "".join(chr(int(t) % self.vocab_size) for t in tokens if isinstance(t, int) or hasattr(t, "__int__"))

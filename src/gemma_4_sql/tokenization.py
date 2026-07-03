"""Tokenization module for processing Text-to-SQL datasets."""

from __future__ import annotations

from gemma_4_sql.backends.lazy_loader import LazyLoader

AutoTokenizer = LazyLoader("transformers").get_module()


class SQLTokenizer:
    """A tokenizer for SQL datasets.

    Wraps a Hugging Face tokenizer (like SentencePiece for Gemma) if available
    and a model_name is provided. Otherwise, falls back to a basic character-level
    encoding scheme.
    """

    def __init__(self, vocab_size: int = 256, model_name: str | None = None) -> None:
        """Initialize the tokenizer.

        Args:
        ----
            vocab_size: Fallback vocabulary size for char-level encoding.
            model_name: Optional Hugging Face model identifier (e.g., 'google/gemma-2b').

        """
        self.vocab_size = vocab_size
        self.model_name = model_name
        self.hf_tokenizer = None
        if self.model_name and AutoTokenizer is not None:
            self.hf_tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of token IDs."""
        if self.hf_tokenizer is not None:
            return self.hf_tokenizer.encode(text, add_special_tokens=False)
        return [ord(c) % self.vocab_size for c in text]

    def decode(self, tokens: list[int]) -> str:
        """Decode a list of token IDs back into a string."""
        if self.hf_tokenizer is not None:
            return self.hf_tokenizer.decode(tokens)
        return "".join(chr(t) for t in tokens)

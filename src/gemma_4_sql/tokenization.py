"""
Tokenization module for processing Text-to-SQL datasets.
"""

from __future__ import annotations

try:
    from transformers import (
        AutoTokenizer,
    )
except ImportError:
    AutoTokenizer = None

class SQLTokenizer:
    """
    A tokenizer for SQL datasets.

    Wraps a Hugging Face tokenizer (like SentencePiece for Gemma) if available
    and a model_name is provided. Otherwise, falls back to a basic character-level
    encoding scheme.
    """

    def __init__(self, vocab_size: int = 256, model_name: str | None = None):
        """
        Initialize the tokenizer.

        Args:
            vocab_size: Fallback vocabulary size for char-level encoding.
            model_name: Optional Hugging Face model identifier (e.g., 'google/gemma-2b').
        """
        self.vocab_size = vocab_size
        self.model_name = model_name
        self._hf_tokenizer = None

        if self.model_name and AutoTokenizer is not None:
            self._hf_tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def encode(self, text: str) -> list[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            text: The input string.

        Returns:
            A list of integer token IDs.
        """
        if self._hf_tokenizer is not None:
            # Type ignore because transformers isn't guaranteed to be fully typed
            return self._hf_tokenizer.encode(text, add_special_tokens=False)
        return [ord(c) % self.vocab_size for c in text]

    def decode(self, tokens: list[int]) -> str:
        """
        Decodes a list of token IDs back into a string.

        Args:
            tokens: The list of token IDs.

        Returns:
            The decoded string.
        """
        if self._hf_tokenizer is not None:
            return self._hf_tokenizer.decode(tokens)
        return "".join(chr(t) for t in tokens)

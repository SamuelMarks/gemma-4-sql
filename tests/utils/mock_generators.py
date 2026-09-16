"""Mock generators for testing language models and dataloaders."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any


class MockCausalModel:
    """Mock causal language model simulating autoregressive generation."""

    def __init__(self, response_text: str = "SELECT * FROM mock_table;") -> None:
        """Initialize MockCausalModel.

        Args:
            response_text: Default SQL text returned upon generation.
        """
        self.response_text = response_text

    def generate(self, *_args: Any, **_kwargs: Any) -> str:
        """Simulate autoregressive text generation.

        Args:
            *_args: Ignored positional arguments.
            **_kwargs: Ignored keyword arguments.

        Returns:
            Mock generated SQL string.
        """
        return self.response_text


class MockIterableDataLoader:
    """Mock dataloader iterating over a predefined list of batch dictionaries."""

    def __init__(self, batches: Sequence[dict[str, Any]]) -> None:
        """Initialize MockIterableDataLoader.

        Args:
            batches: Sequence of batch dictionaries to yield.
        """
        self.batches = list(batches)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Return iterator over batches.

        Returns:
            Batch iterator.
        """
        return iter(self.batches)

    def __len__(self) -> int:
        """Return total batch count.

        Returns:
            Number of batches.
        """
        return len(self.batches)

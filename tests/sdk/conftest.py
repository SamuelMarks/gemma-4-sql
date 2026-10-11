"""Fixtures for SDK tests."""

from collections.abc import Iterator
from typing import Any

import pytest


class MockTokenizer:
    """Standardized mock tokenizer for evaluation and agent tests."""

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        """Mock encode method.

        Args:
            text: Input string to encode.
            **kwargs: Additional tokenizer arguments.

        Returns:
            Mocked token IDs.
        """
        return [1, 2, 3]

    def decode(self, tokens: list[int], **kwargs: Any) -> str:
        """Mock decode method.

        Args:
            tokens: List of token IDs.
            **kwargs: Additional decode arguments.

        Returns:
            Mocked decoded string.
        """
        return "SELECT 1;"


class MockModelEvaluator:
    """Standardized mock model evaluator for evaluation tests."""

    def evaluate(self, model_name: str, dataset: str) -> dict[str, Any]:
        """Mock evaluate method.

        Args:
            model_name: The name of the model.
            dataset: The dataset name.

        Returns:
            Dictionary with evaluation metrics.
        """
        return {"exact_match": 1.0, "execution_accuracy": 1.0}


@pytest.fixture
def mock_tokenizer() -> MockTokenizer:
    """Fixture providing a standardized mock tokenizer.

    Returns:
        MockTokenizer instance.
    """
    return MockTokenizer()


@pytest.fixture
def mock_model_evaluator() -> MockModelEvaluator:
    """Fixture providing a standardized mock model evaluator.

    Returns:
        MockModelEvaluator instance.
    """
    return MockModelEvaluator()


@pytest.fixture
def in_memory_batch_generator() -> Iterator[dict[str, Any]]:
    """Fixture providing a lightweight, typed in-memory batch generator.

    Yields:
        A dummy batch dictionary.
    """
    yield {"inputs": [1, 2], "targets": [2, 3]}

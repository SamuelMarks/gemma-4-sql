# Copyright 2024
"""Tests for the backend approaches."""

from gemma_4_sql.backends.jax import get_trainer as jax_trainer
from gemma_4_sql.backends.keras import get_trainer as keras_trainer
from gemma_4_sql.backends.maxtext import get_trainer as maxtext_trainer
from gemma_4_sql.backends.pytorch import get_trainer as pytorch_trainer


def test_jax() -> None:
    """Test JAX approach.

    Raises:
        AssertionError: Description.

    """
    if jax_trainer() != "jax_trainer":
        raise AssertionError


def test_keras() -> None:
    """Test Keras approach.

    Raises:
        AssertionError: Description.

    """
    if keras_trainer() != "keras_trainer":
        raise AssertionError


def test_maxtext_approach() -> None:
    """Test MaxText approach.

    Raises:
        AssertionError: Description.

    """
    if maxtext_trainer() != "maxtext_trainer":
        raise AssertionError


def test_pytorch() -> None:
    """Test PyTorch approach.

    Raises:
        AssertionError: Description.

    """
    if pytorch_trainer() != "pytorch_trainer":
        raise AssertionError

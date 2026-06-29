"""Tests for the backend approaches."""

from gemma_4_sql.backends.jax import get_trainer as jax_trainer
from gemma_4_sql.backends.keras import get_trainer as keras_trainer
from gemma_4_sql.backends.maxtext import get_trainer as maxtext_trainer
from gemma_4_sql.backends.pytorch import get_trainer as pytorch_trainer


def test_jax() -> None:
    """Test JAX approach."""
    if not jax_trainer() == "jax_trainer":
        raise AssertionError


def test_keras() -> None:
    """Test Keras approach."""
    if not keras_trainer() == "keras_trainer":
        raise AssertionError


def test_maxtext_approach() -> None:
    """Test MaxText approach."""
    if not maxtext_trainer() == "maxtext_trainer":
        raise AssertionError


def test_pytorch() -> None:
    """Test PyTorch approach."""
    if not pytorch_trainer() == "pytorch_trainer":
        raise AssertionError

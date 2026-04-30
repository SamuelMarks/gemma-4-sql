"""
Tests for the backend approaches.
"""

from gemma_4_sql.backends.jax import get_trainer as jax_trainer
from gemma_4_sql.backends.keras import get_trainer as keras_trainer
from gemma_4_sql.backends.maxtext import get_trainer as maxtext_trainer
from gemma_4_sql.backends.pytorch import get_trainer as pytorch_trainer


def test_jax() -> None:
    """Test JAX approach."""
    assert jax_trainer() == "jax_trainer"


def test_keras() -> None:
    """Test Keras approach."""
    assert keras_trainer() == "keras_trainer"


def test_maxtext_approach() -> None:
    """Test MaxText approach."""
    assert maxtext_trainer() == "maxtext_trainer"


def test_pytorch() -> None:
    """Test PyTorch approach."""
    assert pytorch_trainer() == "pytorch_trainer"

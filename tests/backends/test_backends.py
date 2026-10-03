"""Tests for the backend APIs."""

from gemma_4_sql.backends.jax import train_model as jax_train
from gemma_4_sql.backends.keras import train_model as keras_train
from gemma_4_sql.backends.maxtext import train_model as maxtext_train
from gemma_4_sql.backends.mlx import train_model as mlx_train
from gemma_4_sql.backends.pytorch import train_model as pytorch_train


def test_backend_apis_exist() -> None:
    """Test that backends export the train_model API."""
    assert callable(jax_train)
    assert callable(keras_train)
    assert callable(maxtext_train)
    assert callable(pytorch_train)
    assert callable(mlx_train)

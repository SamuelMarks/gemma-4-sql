"""Test that all backends can be imported without collisions."""

import importlib
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def test_import_jax() -> None:
    """Execute function."""
    importlib.import_module("gemma_4_sql.backends.jax")


def test_import_pytorch() -> None:
    """Execute function."""
    importlib.import_module("gemma_4_sql.backends.pytorch")


def test_import_keras() -> None:
    """Execute function."""
    importlib.import_module("gemma_4_sql.backends.keras")


def test_import_maxtext() -> None:
    """Execute function."""
    importlib.import_module("gemma_4_sql.backends.maxtext")


def test_lazy_loader_getters() -> None:
    """Test that lazy loader getters do not crash."""
    get_duckdb = __import__("gemma_4_sql.backends.lazy_loader", fromlist=["get_duckdb"]).get_duckdb
    get_flax_nnx = __import__("gemma_4_sql.backends.lazy_loader", fromlist=["get_flax_nnx"]).get_flax_nnx
    get_jax = __import__("gemma_4_sql.backends.lazy_loader", fromlist=["get_jax"]).get_jax
    get_jnp = __import__("gemma_4_sql.backends.lazy_loader", fromlist=["get_jnp"]).get_jnp
    get_keras = __import__("gemma_4_sql.backends.lazy_loader", fromlist=["get_keras"]).get_keras
    get_maxtext_gemma4 = __import__("gemma_4_sql.backends.lazy_loader", fromlist=["get_maxtext_gemma4"]).get_maxtext_gemma4
    get_mlx = __import__("gemma_4_sql.backends.lazy_loader", fromlist=["get_mlx"]).get_mlx
    get_safetensors = __import__("gemma_4_sql.backends.lazy_loader", fromlist=["get_safetensors"]).get_safetensors
    get_tensorflow = __import__("gemma_4_sql.backends.lazy_loader", fromlist=["get_tensorflow"]).get_tensorflow
    get_torch = __import__("gemma_4_sql.backends.lazy_loader", fromlist=["get_torch"]).get_torch
    get_transformers = __import__("gemma_4_sql.backends.lazy_loader", fromlist=["get_transformers"]).get_transformers
    get_jax()
    get_jnp()
    get_flax_nnx()
    get_tensorflow()
    get_keras()
    get_torch()
    get_mlx()
    get_duckdb()
    get_transformers()
    get_safetensors()
    get_maxtext_gemma4()

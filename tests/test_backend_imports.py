"""Test that all backends can be imported without collisions."""

import importlib
import os

# Prevent JAX and TF from preallocating all memory on MacOS/Metal
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU for import testing


def test_import_jax() -> None:
    importlib.import_module("gemma_4_sql.backends.jax")


def test_import_pytorch() -> None:
    importlib.import_module("gemma_4_sql.backends.pytorch")


def test_import_keras() -> None:
    importlib.import_module("gemma_4_sql.backends.keras")


def test_import_maxtext() -> None:
    importlib.import_module("gemma_4_sql.backends.maxtext")

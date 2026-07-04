# Copyright 2024
"""Provide module docstring."""

import sys
from importlib import import_module
from unittest import mock


def exec_import(mod_name: object, mock_dict: object, func_name: object = None, *args: object, **kwargs: object) -> object:
    """Initialize function exec_import.

    Args:
    ----
    mod_name: Description of mod_name.
    mock_dict: Description of mock_dict.
    func_name: Description of func_name.
    args: Description of args.
    kwargs: Description of kwargs.

    """
    with mock.patch.dict(sys.modules, mock_dict):
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        try:
            mod = import_module(mod_name)
            if func_name:
                getattr(mod, func_name)(*args, **kwargs)
        except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
            pass


def test_missing_jax() -> object:
    """Initialize function test_missing_jax."""
    exec_import("gemma_4_sql.backends.jax.dpo", {"jax.nn": None})
    exec_import("gemma_4_sql.backends.jax.export", {"orbax.checkpoint": None})
    exec_import("gemma_4_sql.backends.jax.inference", {"flax.nnx": None})
    exec_import("gemma_4_sql.backends.jax.quantize", {"jax.numpy": None})
    exec_import("gemma_4_sql.backends.jax.train", {"optax": None})
    exec_import("gemma_4_sql.backends.jax.train", {"flax.nnx": None})


def test_missing_keras() -> object:
    """Initialize function test_missing_keras."""
    exec_import("gemma_4_sql.backends.keras.export", {"keras_nlp": None})
    exec_import("gemma_4_sql.backends.keras.inference", {"tensorflow": None})
    exec_import("gemma_4_sql.backends.keras.inference", {"keras_nlp": None})
    exec_import("gemma_4_sql.backends.keras.train", {"tensorflow": None})
    exec_import("gemma_4_sql.backends.keras.train", {"keras_nlp": None})
    exec_import("gemma_4_sql.backends.keras.train", {"keras_nlp.models": None})


def test_missing_maxtext() -> object:
    """Initialize function test_missing_maxtext."""
    exec_import("gemma_4_sql.backends.maxtext.export", {"orbax.checkpoint": None})
    exec_import("gemma_4_sql.backends.maxtext.export", {"maxtext.models.gemma4": None})
    exec_import("gemma_4_sql.backends.maxtext.train", {"optax": None})


def test_missing_pytorch() -> object:
    """Initialize function test_missing_pytorch."""
    exec_import("gemma_4_sql.backends.pytorch.export", {"transformers.models.gemma4": None})
    exec_import("gemma_4_sql.backends.pytorch.export", {"safetensors.torch": None})
    exec_import("gemma_4_sql.backends.pytorch.train", {"torch.optim": None})


def test_missing_sdk() -> object:
    """Initialize function test_missing_sdk."""
    exec_import("gemma_4_sql.sdk.db_engine", {"psycopg2": None})

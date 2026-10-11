"""Tests for test lazy loader module."""

import sys
from unittest.mock import MagicMock
import pytest

from gemma_4_sql.backends.lazy_loader import (
    LazyLoader,
    catch_optional_imports,
    get_jax,
    get_jnp,
    get_flax_nnx,
    get_torch,
    get_keras,
    get_mlx,
    get_duckdb,
    get_transformers,
    get_safetensors,
    get_maxtext_gemma4,
    get_tensorflow,
    _JAX_LOADER,
    _JNP_LOADER,
    _FLAX_NNX_LOADER,
    _TORCH_LOADER,
    _KERAS_LOADER,
    _MLX_LOADER,
    _DUCKDB_LOADER,
    _TRANSFORMERS_LOADER,
    _SAFETENSORS_LOADER,
    _MAXTEXT_GEMMA4_LOADER,
    _TENSORFLOW_LOADER,
)


def test_catch_optional_imports():
    """Test catch optional imports functionality."""
    with catch_optional_imports():
        import this_module_does_not_exist

    with pytest.raises(ValueError), catch_optional_imports():
        raise ValueError("Should not be caught")


def test_lazy_loader_installed(monkeypatch):
    """Test lazy loader functionality with installed module."""
    monkeypatch.setitem(sys.modules, "my_fake_module", type("Fake", (), {}))
    loader = LazyLoader("my_fake_module")
    assert not loader._loaded
    mod = loader.get_module()
    assert mod is not None
    assert loader._loaded
    assert loader.is_available

    # Check repeated calls return cached
    assert loader.get_module() is mod


def test_lazy_loader_missing():
    """Test lazy loader functionality with missing module."""
    loader = LazyLoader("this_module_does_not_exist_ever")
    assert not loader._loaded
    mod = loader.get_module()
    assert mod is None
    assert loader._loaded
    assert not loader.is_available

    # Check repeated calls return cached None
    assert loader.get_module() is None


def test_lazy_loader_import_error(monkeypatch):
    """Test lazy loader functionality when module raises non-ImportError on import."""
    import importlib

    original_import = importlib.import_module

    def fake_import(name):
        if name == "raising_module":
            raise RuntimeError("Boom!")
        return original_import(name)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    loader = LazyLoader("raising_module")
    with pytest.raises(RuntimeError, match="Boom!"):
        loader.get_module()


@pytest.mark.parametrize(
    "getter, loader_obj, module_name",
    [
        (get_jax, _JAX_LOADER, "jax"),
        (get_jnp, _JNP_LOADER, "jax.numpy"),
        (get_flax_nnx, _FLAX_NNX_LOADER, "flax.nnx"),
        (get_tensorflow, _TENSORFLOW_LOADER, "tensorflow"),
        (get_keras, _KERAS_LOADER, "keras"),
        (get_torch, _TORCH_LOADER, "torch"),
        (get_mlx, _MLX_LOADER, "mlx"),
        (get_duckdb, _DUCKDB_LOADER, "duckdb"),
        (get_transformers, _TRANSFORMERS_LOADER, "transformers"),
        (get_safetensors, _SAFETENSORS_LOADER, "safetensors"),
        (get_maxtext_gemma4, _MAXTEXT_GEMMA4_LOADER, "maxtext.models.gemma4"),
    ],
)
def test_lazy_loader_getters(getter, loader_obj, module_name, monkeypatch):
    """Test all specific getters under available and missing conditions."""
    loader_obj._loaded = False
    loader_obj._module = None

    # Missing condition
    monkeypatch.setitem(sys.modules, module_name, None)

    import importlib

    original_import = importlib.import_module

    def fake_import(name):
        if name == module_name:
            raise ImportError("Fake missing")
        return original_import(name)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    assert getter() is None

    loader_obj._loaded = False
    loader_obj._module = None

    # Available condition
    fake_mod = type("Fake", (), {})
    monkeypatch.setitem(sys.modules, module_name, fake_mod)

    def fake_import_ok(name):
        if name == module_name:
            return fake_mod
        return original_import(name)

    monkeypatch.setattr(importlib, "import_module", fake_import_ok)

    assert getter() is fake_mod

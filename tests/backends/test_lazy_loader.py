"""Unit tests for LazyLoader and optional dependency resolution across backends."""

from __future__ import annotations

import importlib
import sys
import types
from collections.abc import Callable
from typing import NoReturn

import pytest

from gemma_4_sql.backends.lazy_loader import (
    _DUCKDB_LOADER,
    _FLAX_NNX_LOADER,
    _JAX_LOADER,
    _JNP_LOADER,
    _KERAS_LOADER,
    _MAXTEXT_GEMMA4_LOADER,
    _MLX_LOADER,
    _SAFETENSORS_LOADER,
    _TENSORFLOW_LOADER,
    _TORCH_LOADER,
    _TRANSFORMERS_LOADER,
    LazyLoader,
    catch_optional_imports,
    get_duckdb,
    get_flax_nnx,
    get_jax,
    get_jnp,
    get_keras,
    get_maxtext_gemma4,
    get_mlx,
    get_safetensors,
    get_tensorflow,
    get_torch,
    get_transformers,
    optional_attr,
    optional_import,
    override_lazy_module,
    reset_all_lazy_loaders,
)


def test_catch_optional_imports() -> None:
    """Verify catch_optional_imports suppresses ImportError/ModuleNotFoundError and propagates ValueError."""
    with catch_optional_imports():
        raise ModuleNotFoundError("optional_dep_xyz")

    with pytest.raises(ValueError, match="Should not be caught"), catch_optional_imports():
        raise ValueError("Should not be caught")


def test_lazy_loader_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify LazyLoader loads and caches an installed module."""
    fake_mod = types.ModuleType("my_fake_module")
    monkeypatch.setitem(sys.modules, "my_fake_module", fake_mod)
    loader: LazyLoader[types.ModuleType] = LazyLoader("my_fake_module")
    assert not loader._loaded
    mod = loader.get_module()
    assert mod is fake_mod
    assert loader._loaded
    assert loader.is_available
    assert loader.get_module() is fake_mod


def test_lazy_loader_missing() -> None:
    """Verify LazyLoader returns None and caches result when module is missing."""
    loader: LazyLoader[types.ModuleType] = LazyLoader("this_module_does_not_exist_ever")
    assert not loader._loaded
    mod = loader.get_module()
    assert mod is None
    assert loader._loaded
    assert not loader.is_available
    assert loader.get_module() is None


def test_lazy_loader_submodule_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify LazyLoader resolves submodules and optional_import / optional_attr work."""
    fake_pkg = types.ModuleType("fake_pkg_test")
    fake_sub = types.ModuleType("fake_pkg_test.sub")
    monkeypatch.setattr(fake_sub, "val", "ok", raising=False)
    monkeypatch.setitem(sys.modules, "fake_pkg_test", fake_pkg)
    monkeypatch.setitem(sys.modules, "fake_pkg_test.sub", fake_sub)

    assert optional_import("fake_pkg_test.sub") is fake_sub
    assert optional_attr("fake_pkg_test.sub", "val") == "ok"
    assert optional_attr("fake_pkg_test.sub", "nonexistent") is None
    assert optional_attr("nonexistent_module_xyz", "val") is None

    class RaisingAttrModule(types.ModuleType):
        """Module whose __getattr__ raises ImportError (simulating transformers lazy objects)."""

        def __getattr__(self, name: str) -> NoReturn:
            """Raise ImportError."""
            _ = name
            raise ImportError("lazy attr failed")

    monkeypatch.setitem(sys.modules, "raising_attr_mod", RaisingAttrModule("raising_attr_mod"))
    assert optional_attr("raising_attr_mod", "some_cls") is None


def test_lazy_loader_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify LazyLoader propagates non-ImportError/AttributeError exceptions."""

    def fake_import(_name: str, _package: str | None = None) -> types.ModuleType:
        """Raise RuntimeError unconditionally."""
        raise RuntimeError("Boom!")

    monkeypatch.setattr(importlib, "import_module", fake_import)

    loader: LazyLoader[types.ModuleType] = LazyLoader("raising_module")
    with pytest.raises(RuntimeError, match="Boom!"):
        loader.get_module()


@pytest.mark.parametrize(
    ("getter", "loader_obj", "module_name"),
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
def test_lazy_loader_getters(
    getter: Callable[[], types.ModuleType | None],
    loader_obj: LazyLoader[types.ModuleType],
    module_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify all specific module getters under both missing and available conditions."""
    loader_obj._loaded = False
    loader_obj._module = None
    monkeypatch.delitem(sys.modules, module_name, raising=False)

    def fake_import_missing(_name: str, _package: str | None = None) -> types.ModuleType:
        """Simulate missing module."""
        raise ImportError("Fake missing")

    monkeypatch.setattr(importlib, "import_module", fake_import_missing)
    assert getter() is None

    loader_obj._loaded = False
    loader_obj._module = None

    fake_mod = types.ModuleType(module_name)
    monkeypatch.setattr(importlib, "import_module", lambda _name, _package=None: fake_mod)  # pyright: ignore[reportUnknownLambdaType]
    assert getter() is fake_mod


def test_reset_all_and_override_lazy_module() -> None:
    """Verify reset_all_lazy_loaders and override_lazy_module context manager."""
    fake_mod = types.ModuleType("custom_overridden_mod")
    with override_lazy_module("custom_overridden_mod", fake_mod):
        assert _JAX_LOADER.module_name == "jax"
        from gemma_4_sql.backends.lazy_loader import _LOADER_REGISTRY

        assert _LOADER_REGISTRY["custom_overridden_mod"][0].get_module() is fake_mod

    with override_lazy_module("jax", fake_mod):
        assert get_jax() is fake_mod

    reset_all_lazy_loaders()
    assert not _JAX_LOADER._loaded

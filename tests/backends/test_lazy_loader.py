"""Unit tests for flyweight LazyLoader, typed protocol validation, DI importers, and optional symbol resolution."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Generator
from typing import Protocol, runtime_checkable

import pytest

from gemma_4_sql.backends.lazy_loader import (
    _DUCKDB_LOADER,
    _FLAX_NNX_LOADER,
    _JAX_LOADER,
    _JNP_LOADER,
    _KERAS_LOADER,
    _LOADER_REGISTRY,
    _MAXTEXT_GEMMA4_LOADER,
    _MLX_LOADER,
    _SAFETENSORS_LOADER,
    _TENSORFLOW_LOADER,
    _TORCH_LOADER,
    _TRANSFORMERS_LOADER,
    LazyLoader,
    ModuleNamespaceProtocol,
    SubmoduleNamespaceProtocol,
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
    import_optional_submodule,
    import_optional_symbol,
    optional_import,
    override_lazy_module,
    reset_all_lazy_loaders,
)
from tests.utils.test_utils import FakeModuleNamespace


@runtime_checkable
class _VersionedModuleProtocol(Protocol):
    """Protocol requiring an `api_version` string attribute."""

    @property
    def api_version(self) -> str:
        """Provide module API version string."""
        ...


class _VersionedFakeModule(FakeModuleNamespace):
    """Typed fake module exposing `api_version`."""

    api_version: str

    def __init__(self, name: str, api_version: str) -> None:
        """Initialize versioned fake module.

        Args:
            name: Module name.
            api_version: Semantic version string.

        """
        super().__init__(name)
        self.api_version = api_version


class _FakePkgRoot(FakeModuleNamespace):
    """Typed package module exposing `inline_sub` and `not_a_mod`."""

    inline_sub: FakeModuleNamespace
    not_a_mod: int

    def __init__(self, name: str, inline_sub: FakeModuleNamespace) -> None:
        """Initialize fake package root.

        Args:
            name: Package name.
            inline_sub: Embedded submodule double.

        """
        super().__init__(name)
        self.inline_sub = inline_sub
        self.not_a_mod = 123


class _FakeSubmoduleWithSymbols(FakeModuleNamespace):
    """Typed submodule exposing `val` and `num` attributes."""

    val: str
    num: int

    def __init__(self, name: str, val: str, num: int) -> None:
        """Initialize fake submodule with typed attributes.

        Args:
            name: Submodule name.
            val: String symbol value.
            num: Integer symbol value.

        """
        super().__init__(name)
        self.val = val
        self.num = num


class _FakeReexportSubmodule(FakeModuleNamespace):
    """Typed re-export submodule exposing `LazyCls` in `vars(...)`."""

    LazyCls: type[str]

    def __init__(self, name: str) -> None:
        """Initialize re-export submodule with `LazyCls = str`.

        Args:
            name: Dotted submodule name.

        """
        super().__init__(name)
        self.LazyCls = str


@pytest.fixture(autouse=True)
def _restore_loaders_after_test() -> Generator[None, None, None]:
    """Reset all `LazyLoader` instances after each test to preserve cross-test isolation.

    Yields:
        None during test execution.

    """
    yield
    reset_all_lazy_loaders()


def test_catch_optional_imports() -> None:
    """Verify `catch_optional_imports` suppresses `ImportError` and propagates non-import exceptions.

    Raises:
        ModuleNotFoundError: Caught inside `catch_optional_imports`.
        ValueError: Propagated outside `catch_optional_imports` and asserted by `pytest.raises`.

    """
    with catch_optional_imports():
        raise ModuleNotFoundError("optional_dep_xyz")

    with pytest.raises(ValueError, match="Should not be caught"), catch_optional_imports():
        raise ValueError("Should not be caught")


def test_lazy_loader_flyweight_and_protocol_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `LazyLoader` deduplicates flyweights per `module_name` and validates `@runtime_checkable` protocols."""
    fake_mod = _VersionedFakeModule("my_fake_module", "1.0")
    monkeypatch.setitem(sys.modules, "my_fake_module", fake_mod)

    loader1 = LazyLoader("my_fake_module")
    loader2 = LazyLoader("my_fake_module")
    assert loader1 is loader2
    assert not loader1._loaded
    mod = loader1.get_module()
    assert mod is fake_mod
    assert loader1._module is fake_mod
    assert loader1._loaded
    assert loader1.is_available
    assert loader1.get_module() is fake_mod

    # Typed protocol view shares the slot and validates `isinstance(mod, _VersionedModuleProtocol)`
    typed_loader = LazyLoader.for_protocol("my_fake_module", _VersionedModuleProtocol)
    typed_mod = typed_loader.get_module()
    assert typed_mod is not None
    assert typed_mod.api_version == "1.0"

    # Custom DI importer hook on both base and typed loaders without mutating global `importlib`
    di_mod = _VersionedFakeModule("my_di_module", "3.0")
    di_base_loader = LazyLoader("my_di_module", importer=FakeModuleNamespace)
    di_loaded = di_base_loader.get_module()
    assert di_loaded is not None
    assert di_loaded.__name__ == "my_di_module"

    di_typed_loader = LazyLoader.for_protocol(
        "my_di_typed_module",
        _VersionedModuleProtocol,
        importer=lambda _n: di_mod,
    )
    di_typed_loaded = di_typed_loader.get_module()
    assert di_typed_loaded is di_mod

    # Typed loader on a fresh module name also populates `_LOADER_REGISTRY`
    fresh_mod = _VersionedFakeModule("my_fresh_typed_mod", "2.0")
    monkeypatch.setitem(sys.modules, "my_fresh_typed_mod", fresh_mod)
    fresh_typed = LazyLoader.for_protocol("my_fresh_typed_mod", _VersionedModuleProtocol)
    assert "my_fresh_typed_mod" in _LOADER_REGISTRY
    assert fresh_typed.get_module() is not None
    assert optional_import("my_fresh_typed_mod", _VersionedModuleProtocol) is not None

    # Module missing required protocol attribute returns `None` from typed loader!
    plain_mod = FakeModuleNamespace("plain_no_version_mod")
    monkeypatch.setitem(sys.modules, "plain_no_version_mod", plain_mod)
    mismatch_loader = LazyLoader.for_protocol("plain_no_version_mod", _VersionedModuleProtocol)
    assert mismatch_loader.get_module() is None
    assert optional_import("plain_no_version_mod", _VersionedModuleProtocol) is None

    # Overridden typed optional_import branch
    with override_lazy_module("my_fresh_typed_mod", fresh_mod):
        assert optional_import("my_fresh_typed_mod", _VersionedModuleProtocol) is fresh_mod
    monkeypatch.setitem(sys.modules, "my_fresh_typed_mod", None)
    assert optional_import("my_fresh_typed_mod", _VersionedModuleProtocol) is None

    # Non-sys.modules typed optional_import branch via custom importer on underlying slot
    non_sys_mod = _VersionedFakeModule("non_sys_typed_mod", "4.0")
    _ = LazyLoader("non_sys_typed_mod", importer=lambda _n: non_sys_mod).get_module()
    assert optional_import("non_sys_typed_mod", _VersionedModuleProtocol) is non_sys_mod


def test_lazy_loader_missing() -> None:
    """Verify `LazyLoader` returns `None` and caches result when module is missing."""
    loader = LazyLoader("this_module_does_not_exist_ever")
    loader.reset()
    assert not loader._loaded
    mod = loader.get_module()
    assert mod is None
    assert loader._loaded
    assert not loader.is_available
    assert loader.get_module() is None
    assert optional_import("this_module_does_not_exist_ever") is None


def test_submodule_and_symbol_resolution_without_getattr(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `import_optional_submodule` and `import_optional_symbol` resolve `vars(mod)` and submodules."""
    fake_sub = _FakeSubmoduleWithSymbols("fake_pkg_test.sub", "ok", 42)
    fake_pkg = _FakePkgRoot("fake_pkg_test", fake_sub)
    fake_child = FakeModuleNamespace("fake_pkg_test.child")
    monkeypatch.setitem(sys.modules, "fake_pkg_test", fake_pkg)
    monkeypatch.setitem(sys.modules, "fake_pkg_test.sub", fake_sub)
    monkeypatch.setitem(sys.modules, "fake_pkg_test.child", fake_child)

    assert optional_import("fake_pkg_test.sub") is fake_sub
    assert import_optional_submodule("fake_pkg_test", "inline_sub") is fake_sub
    assert import_optional_submodule("fake_pkg_test", "child") is fake_child
    assert import_optional_submodule("fake_pkg_test", "not_a_mod") is None
    assert import_optional_submodule("nonexistent_module_xyz", "sub") is None

    assert import_optional_symbol("fake_pkg_test.sub", "val", str) == "ok"
    # Runtime `isinstance` check rejects mismatched types (`"ok"` is not `int`)!
    assert import_optional_symbol("fake_pkg_test.sub", "val", int) is None
    assert import_optional_symbol("fake_pkg_test.sub", "nonexistent", str) is None
    assert import_optional_symbol("nonexistent_module_xyz", "val", str) is None
    # Submodule fallback when `child` is not in `vars(fake_pkg)` and `protocol` is `SubmoduleNamespaceProtocol`
    assert import_optional_symbol("fake_pkg_test", "child", SubmoduleNamespaceProtocol) is fake_child

    # Canonical lazy re-export resolution (`_LAZY_REEXPORT_SUBMODULES`)
    fake_tf = FakeModuleNamespace("fake_reexport_root")
    fake_reexport_sub = _FakeReexportSubmodule("fake_reexport_root.auto_mod")
    monkeypatch.setitem(sys.modules, "fake_reexport_root", fake_tf)
    monkeypatch.setitem(sys.modules, "fake_reexport_root.auto_mod", fake_reexport_sub)
    from gemma_4_sql.backends import lazy_loader as ll_mod

    monkeypatch.setitem(
        ll_mod._LAZY_REEXPORT_SUBMODULES, ("fake_reexport_root", "LazyCls"), "fake_reexport_root.auto_mod"
    )
    assert import_optional_symbol("fake_reexport_root", "LazyCls", type) is str
    assert import_optional_symbol("fake_reexport_root", "LazyCls", int) is None
    monkeypatch.setitem(sys.modules, "fake_reexport_root.auto_mod", None)
    assert import_optional_symbol("fake_reexport_root", "LazyCls", type) is None

    # Overridden parent module does NOT fall through to `sys.modules` submodules
    override_empty = FakeModuleNamespace("fake_pkg_test")
    with override_lazy_module("fake_pkg_test", override_empty):
        assert import_optional_submodule("fake_pkg_test", "child") is None
        assert import_optional_symbol("fake_pkg_test", "child", SubmoduleNamespaceProtocol) is None


def test_lazy_loader_import_error() -> None:
    """Verify `LazyLoader` propagates non-`ImportError`/`AttributeError` exceptions via DI `importer`."""

    def fake_import(_name: str) -> ModuleNamespaceProtocol:
        """Raise `RuntimeError` unconditionally.

        Args:
            _name: Module name.

        Raises:
            RuntimeError: Always raised.

        """
        raise RuntimeError("Boom!")

    loader = LazyLoader("raising_module", importer=fake_import)
    loader.reset()
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
    getter: Callable[[], ModuleNamespaceProtocol | None],
    loader_obj: LazyLoader[ModuleNamespaceProtocol],
    module_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify all specific module getters under missing, `sys.modules[name] = None`, and available conditions."""
    loader_obj._loaded = False
    loader_obj._module = None
    monkeypatch.setitem(sys.modules, module_name, None)
    assert getter() is None
    assert optional_import(module_name) is None

    monkeypatch.delitem(sys.modules, module_name, raising=False)

    def fake_import_missing(_name: str, _package: str | None = None) -> ModuleNamespaceProtocol:
        """Simulate missing module.

        Args:
            _name: Module name.
            _package: Optional package context.

        Raises:
            ImportError: Always raised.

        """
        raise ImportError("Fake missing")

    monkeypatch.setattr(importlib, "import_module", fake_import_missing)
    assert getter() is None

    loader_obj._loaded = False
    loader_obj._module = None

    fake_mod = FakeModuleNamespace(module_name)

    def _return_fake_mod(_name: str, _package: str | None = None) -> ModuleNamespaceProtocol:
        """Return `fake_mod`.

        Args:
            _name: Module name.
            _package: Optional package context.

        Returns:
            Injected `fake_mod` instance.

        """
        _ = (_name, _package)
        return fake_mod

    monkeypatch.setattr(importlib, "import_module", _return_fake_mod)
    assert getter() is fake_mod


def test_reset_all_and_override_lazy_module() -> None:
    """Verify `reset_all_lazy_loaders` and `override_lazy_module` context manager."""
    fake_mod = FakeModuleNamespace("custom_overridden_mod")
    with override_lazy_module("custom_overridden_mod", fake_mod):
        assert _JAX_LOADER.module_name == "jax"
        assert _LOADER_REGISTRY["custom_overridden_mod"].get_module() is fake_mod

    with override_lazy_module("jax", fake_mod):
        assert get_jax() is fake_mod

    with override_lazy_module("jax", None):
        assert get_jax() is None

    reset_all_lazy_loaders()
    assert not _JAX_LOADER._loaded

"""Tests for the SDK backend registry module using explicit typed protocol doubles and DI."""

from __future__ import annotations

from collections.abc import Callable, Generator

import pytest

import gemma_4_sql.sdk.registry as mod
from gemma_4_sql.backends.lazy_loader import ModuleNamespaceProtocol
from gemma_4_sql.exceptions import ConfigurationError
from gemma_4_sql.sdk.protocols import BackendProtocol
from gemma_4_sql.sdk.registry import (
    EntryPointsCollection,
    clear_backend_cache,
    get_backend,
    override_backend,
    validate_backend_name,
)
from tests.utils.test_utils import FakeModuleNamespace, RecordingBackendDouble


class _TypedEntryPoint:
    """Typed `EntryPointProtocol` test double."""

    def __init__(self, name: str, module_name: str = "mock_module") -> None:
        """Initialize entry point with `name` and loaded `FakeModuleNamespace`.

        Args:
            name: Entry-point identifier.
            module_name: Target module name returned by `load()`.

        """
        self.name = name
        self._module = FakeModuleNamespace(module_name)
        self.load_calls = 0

    def load(self) -> ModuleNamespaceProtocol:
        """Increment `load_calls` and return `self._module`.

        Returns:
            Loaded module namespace double.

        """
        self.load_calls += 1
        return self._module


class _TypedSelectableEntryPoints:
    """Typed `SelectableEntryPointsProtocol` test double."""

    def __init__(self, entries: list[_TypedEntryPoint]) -> None:
        """Store entries returned by `select`.

        Args:
            entries: Entry-point descriptors to return.

        """
        self._entries = entries
        self.selected_groups: list[str] = []

    def select(self, *, group: str) -> list[_TypedEntryPoint]:
        """Record `group` and return `self._entries`.

        Args:
            group: Entry-point group identifier.

        Returns:
            List of matching `_TypedEntryPoint` instances.

        """
        self.selected_groups.append(group)
        return self._entries


class _TypedAdapterModule(FakeModuleNamespace):
    """Typed `.adapter` module double exposing `ADAPTER_CLASS`."""

    def __init__(
        self,
        name: str,
        adapter_cls: Callable[[], BackendProtocol | FakeModuleNamespace] | None,
    ) -> None:
        """Initialize adapter module namespace with `ADAPTER_CLASS`.

        Args:
            name: Dotted adapter module name.
            adapter_cls: Adapter constructor or `None`.

        """
        super().__init__(name)
        vars(self)["ADAPTER_CLASS"] = adapter_cls


@pytest.fixture(autouse=True)
def _clean_registry_caches() -> Generator[None, None, None]:
    """Clear registry caches before and after each test in `test_registry.py`.

    Yields:
        None during test execution.

    """
    clear_backend_cache()
    yield
    clear_backend_cache()


def test_registry_fallback_and_validate_backend_name() -> None:
    """Verify unknown backend names raise `ConfigurationError` and `override_backend` restores state."""
    mod.ENTRY_POINTS.clear()
    mod._BACKEND_CACHE.clear()

    validate_backend_name("jax")
    with pytest.raises(ConfigurationError, match="Unknown backend: nonexistent"):
        validate_backend_name("nonexistent")

    mod.ENTRY_POINTS.clear()
    with pytest.raises(ConfigurationError, match="Unknown backend: nonexistent"):
        get_backend("nonexistent")

    # Cached backend fast-path in `validate_backend_name` and `override_backend` restoration
    real_pt = get_backend("pytorch_native")
    validate_backend_name("pytorch_native")
    injected = RecordingBackendDouble("pytorch_native")
    with override_backend("pytorch_native", injected) as active_pt:
        assert active_pt is injected
        assert get_backend("pytorch_native") is injected
    assert get_backend("pytorch_native") is real_pt

    temp_double = RecordingBackendDouble("ephemeral_backend")
    with override_backend("ephemeral_backend", temp_double):
        assert get_backend("ephemeral_backend") is temp_double
    assert "ephemeral_backend" not in mod._BACKEND_CACHE


def test_registry_missing_adapter_module() -> None:
    """Verify missing `.adapter` submodule raises `ConfigurationError` via injected `module_loader`."""
    fake_ep = _TypedEntryPoint("mock_backend")

    def _ep_provider() -> EntryPointsCollection:
        """Return mapping with `fake_ep`."""
        return {"gemma_4_sql.backends": [fake_ep]}

    def _failing_loader(name: str) -> ModuleNamespaceProtocol:
        """Raise `ImportError` for `mock_module.adapter`.

        Args:
            name: Dotted module name.

        Returns:
            Loaded module namespace when not `mock_module.adapter`.

        Raises:
            ImportError: When `name` is `mock_module.adapter`.

        """
        if name == "mock_module.adapter":
            raise ImportError("Mock missing adapter")
        return FakeModuleNamespace(name)

    assert _failing_loader("other.mod").__name__ == "other.mod"
    with pytest.raises(ConfigurationError, match=r"Backend mock_backend is missing the adapter module\."):
        get_backend("mock_backend", entry_points_provider=_ep_provider, module_loader=_failing_loader)


def test_registry_missing_adapter_class() -> None:
    """Verify missing `ADAPTER_CLASS` attribute raises `ConfigurationError`."""
    fake_ep = _TypedEntryPoint("mock_backend")
    fake_adapter = FakeModuleNamespace("mock_module.adapter")

    def _ep_provider() -> EntryPointsCollection:
        """Return mapping with `fake_ep`."""
        return {"gemma_4_sql.backends": [fake_ep]}

    def _loader(name: str) -> ModuleNamespaceProtocol:
        """Return `fake_adapter` without `ADAPTER_CLASS`."""
        _ = name
        return fake_adapter

    with pytest.raises(
        ConfigurationError,
        match=r"Adapter class Mock_backendBackend not found in mock_module\.adapter",
    ):
        get_backend("mock_backend", entry_points_provider=_ep_provider, module_loader=_loader)


def test_registry_dict_and_select_entry_points_and_caching() -> None:
    """Verify dict entry points, `.select(group=...)` entry points, and `_BACKEND_CACHE` reuse."""
    fake_ep = _TypedEntryPoint("mock_backend")
    adapter_mod = _TypedAdapterModule("mock_module.adapter", RecordingBackendDouble)
    ep_calls = 0

    def _dict_ep_provider() -> EntryPointsCollection:
        """Count discovery calls and return dict-shaped entry points.

        Returns:
            Mapping of backend group name to `[fake_ep]`.

        """
        nonlocal ep_calls
        ep_calls += 1
        return {"gemma_4_sql.backends": [fake_ep]}

    def _loader(name: str) -> ModuleNamespaceProtocol:
        """Return `adapter_mod` for `mock_module.adapter`."""
        _ = name
        return adapter_mod

    b1 = get_backend("mock_backend", entry_points_provider=_dict_ep_provider, module_loader=_loader)
    b2 = get_backend("mock_backend", entry_points_provider=_dict_ep_provider, module_loader=_loader)
    assert isinstance(b1, BackendProtocol)
    assert b1 is b2
    assert fake_ep.load_calls == 1
    assert ep_calls == 1

    clear_backend_cache()
    sel_ep = _TypedEntryPoint("select_backend")
    sel_eps = _TypedSelectableEntryPoints([sel_ep])
    backend = get_backend("select_backend", entry_points_provider=lambda: sel_eps, module_loader=_loader)
    assert isinstance(backend, RecordingBackendDouble)
    assert sel_eps.selected_groups == ["gemma_4_sql.backends"]


def test_registry_invalid_protocol_instance() -> None:
    """Verify `get_backend` raises `TypeError` when adapter instance does not satisfy `BackendProtocol`."""
    fake_ep = _TypedEntryPoint("invalid_proto")

    def _make_non_backend() -> FakeModuleNamespace:
        """Return a non-`BackendProtocol` object."""
        return FakeModuleNamespace("not_a_backend")

    adapter_mod = _TypedAdapterModule("mock_module.adapter", _make_non_backend)

    with pytest.raises(TypeError, match="does not satisfy BackendProtocol"):
        get_backend(
            "invalid_proto",
            entry_points_provider=lambda: {"gemma_4_sql.backends": [fake_ep]},
            module_loader=lambda _name: adapter_mod,
        )


def test_tpu_comparison_backends_and_aliases_registered() -> None:
    """Verify all 7 `TPU_COMPARISON_BACKENDS` and short Keras aliases resolve through `get_backend`."""
    from gemma_4_sql.backends.base_adapter import ModuleDelegatedBackend
    from gemma_4_sql.constants import KERAS_BACKEND_ALIAS_MAP, TPU_COMPARISON_BACKENDS

    assert len(TPU_COMPARISON_BACKENDS) == 7
    assert KERAS_BACKEND_ALIAS_MAP["keras_jax"] == "jax"
    assert KERAS_BACKEND_ALIAS_MAP["keras_tensorflow"] == "tensorflow"
    assert KERAS_BACKEND_ALIAS_MAP["keras_tf"] == "tensorflow"
    assert KERAS_BACKEND_ALIAS_MAP["keras_pytorch"] == "torch"
    assert KERAS_BACKEND_ALIAS_MAP["keras_torch"] == "torch"

    for alias in (*TPU_COMPARISON_BACKENDS, "keras_tf", "keras_torch", "pytorch_hf"):
        validate_backend_name(alias)
        backend_inst = get_backend(alias)
        assert isinstance(backend_inst, BackendProtocol)
        assert isinstance(backend_inst, ModuleDelegatedBackend)
        if alias == "maxtext":
            assert backend_inst.backend_alias is None
            assert backend_inst.get_trainer(include_alias=True) == "maxtext_trainer"
        else:
            assert backend_inst.backend_alias == alias
            assert backend_inst.get_trainer().endswith("_trainer")
            assert backend_inst.get_trainer(include_alias=True) == f"{alias}_trainer"

    for canonical in ("jax", "keras", "pytorch"):
        base_inst = get_backend(canonical)
        assert isinstance(base_inst, ModuleDelegatedBackend)
        assert base_inst.backend_alias is None
        assert base_inst.get_trainer(include_alias=True) == f"{canonical}_trainer"

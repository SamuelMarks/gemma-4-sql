"""Centralized thread-safe lazy loader for optional backend dependencies without `getattr`."""

from __future__ import annotations

import importlib
import logging
import sys
import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Generic, Protocol, TypeVar, overload, runtime_checkable

logger = logging.getLogger(__name__)
OPTIONAL_IMPORT_ERRORS = (ImportError,)
_LOADER_LOCK = threading.RLock()

_LAZY_REEXPORT_SUBMODULES: dict[tuple[str, str], str] = {
    ("transformers", "AutoModelForCausalLM"): "transformers.models.auto.modeling_auto",
    ("transformers", "AutoTokenizer"): "transformers.models.auto.tokenization_auto",
    ("transformers", "BitsAndBytesConfig"): "transformers.utils.quantization_config",
}


@runtime_checkable
class ModuleNamespaceProtocol(Protocol):
    """Structural protocol satisfied by Python modules and typed module doubles.

    Attributes:
        __name__: Fully-qualified dotted module name.

    """

    __name__: str


@runtime_checkable
class SubmoduleNamespaceProtocol(ModuleNamespaceProtocol, Protocol):
    """Structural protocol satisfied by Python submodule objects (distinguishing modules from classes).

    Attributes:
        __package__: Enclosing package name or `None` for top-level modules.

    """

    __package__: str | None


type ModuleImporterCallable = Callable[[str], ModuleNamespaceProtocol]


@contextmanager
def catch_optional_imports() -> Generator[None, None, None]:
    """Context manager to gracefully catch missing optional backend dependencies.

    Yields:
        None when entering the guarded optional import block.

    """
    try:
        yield
    except OPTIONAL_IMPORT_ERRORS:
        pass


T_co = TypeVar("T_co", covariant=True)
TSymbol = TypeVar("TSymbol")


class _ModuleSlot:
    """Shared per-module state backing all `LazyLoader` views for a `module_name`."""

    module_name: str
    module: ModuleNamespaceProtocol | None
    loaded: bool
    overridden: bool

    def __init__(self, module_name: str) -> None:
        """Initialize the shared module slot.

        Args:
            module_name: Dotted module name string.

        """
        self.module_name = module_name
        self.module = None
        self.loaded = False
        self.overridden = False


_MODULE_SLOTS: dict[str, _ModuleSlot] = {}
_LOADER_REGISTRY: dict[str, LazyLoader[ModuleNamespaceProtocol]] = {}


def _get_or_create_slot(module_name: str) -> _ModuleSlot:
    """Return the deduplicated `_ModuleSlot` for `module_name`.

    Args:
        module_name: Dotted module name string.

    Returns:
        Singleton `_ModuleSlot` instance for `module_name`.

    """
    with _LOADER_LOCK:
        slot = _MODULE_SLOTS.get(module_name)
        if slot is None:
            slot = _ModuleSlot(module_name)
            _MODULE_SLOTS[module_name] = slot
        return slot


def _default_import_module(module_name: str) -> ModuleNamespaceProtocol:
    """Import `module_name` via `importlib.import_module` as a `ModuleNamespaceProtocol`.

    Args:
        module_name: Dotted module name string.

    Returns:
        Imported module namespace.

    """
    return importlib.import_module(module_name)


def _accept_any_module(mod: ModuleNamespaceProtocol) -> ModuleNamespaceProtocol:
    """Identity validator accepting any `ModuleNamespaceProtocol` instance.

    Args:
        mod: Loaded module namespace.

    Returns:
        Unmodified `mod` instance.

    """
    return mod


def _make_protocol_validator(
    protocol: type[TSymbol],
) -> Callable[[ModuleNamespaceProtocol], TSymbol | None]:
    """Create a runtime `isinstance` validator narrowing a module namespace to `TSymbol`.

    Args:
        protocol: `@runtime_checkable` protocol or runtime class.

    Returns:
        Callable returning `TSymbol` when `isinstance(mod, protocol)` holds, else `None`.

    """

    def _validate(mod: ModuleNamespaceProtocol) -> TSymbol | None:
        """Validate `mod` against `protocol`.

        Args:
            mod: Candidate module namespace.

        Returns:
            Narrowed `TSymbol` or `None`.

        """
        return mod if isinstance(mod, protocol) else None

    return _validate


class LazyLoader(Generic[T_co]):
    """Thread-safe flyweight lazy loader for an optional module dependency.

    Validates imported modules via `_validator` (`isinstance(..., protocol)`)
    and deduplicates default module loaders per `module_name` in `_LOADER_REGISTRY`.

    Attributes:
        module_name: Dotted module name string to import lazily.

    """

    module_name: str
    _slot: _ModuleSlot
    _validator: Callable[[ModuleNamespaceProtocol], T_co | None]
    _importer: ModuleImporterCallable | None

    @staticmethod
    def for_protocol(
        module_name: str,
        protocol: type[TSymbol],
        *,
        importer: ModuleImporterCallable | None = None,
    ) -> LazyLoader[TSymbol]:
        """Construct a protocol-validated `LazyLoader[TSymbol]` sharing `module_name`'s slot.

        Args:
            module_name: Dotted module name string to import lazily.
            protocol: `@runtime_checkable` protocol or type to validate against.
            importer: Optional custom module importer hook for dependency injection.

        Returns:
            Protocol-validated `LazyLoader[TSymbol]` instance.

        """
        return LazyLoader(module_name, protocol, importer=importer)

    @overload
    def __new__(
        cls,
        module_name: str,
        protocol: None = None,
        *,
        importer: ModuleImporterCallable | None = None,
    ) -> LazyLoader[ModuleNamespaceProtocol]:
        """Construct or retrieve a `ModuleNamespaceProtocol` lazy loader."""

    @overload
    def __new__(
        cls,
        module_name: str,
        protocol: type[TSymbol],
        *,
        importer: ModuleImporterCallable | None = None,
    ) -> LazyLoader[TSymbol]:
        """Construct a protocol-validated `LazyLoader[TSymbol]` instance."""

    def __new__(
        cls,
        module_name: str,
        protocol: type[TSymbol] | None = None,
        *,
        importer: ModuleImporterCallable | None = None,
    ) -> LazyLoader[TSymbol] | LazyLoader[ModuleNamespaceProtocol]:
        """Return the deduplicated flyweight or protocol-bound `LazyLoader` for `module_name`.

        Args:
            module_name: Dotted module name string to import lazily.
            protocol: Optional `@runtime_checkable` protocol to validate against.
            importer: Optional custom module importer callable for dependency injection.

        Returns:
            Deduplicated flyweight or protocol-validated `LazyLoader` instance.

        """
        _ = cls
        with _LOADER_LOCK:
            slot = _get_or_create_slot(module_name)
            base = _LOADER_REGISTRY.get(module_name)
            if base is None:
                base = _AllocatedLazyLoader(module_name, slot, _accept_any_module, None)
                _LOADER_REGISTRY[module_name] = base
            if protocol is None and importer is None:
                return base
            if protocol is None:
                return _AllocatedLazyLoader(module_name, slot, _accept_any_module, importer)
            return _AllocatedLazyLoader(module_name, slot, _make_protocol_validator(protocol), importer)

    @property
    def _loaded(self) -> bool:
        """Flag indicating whether the underlying module slot has attempted loading."""
        with _LOADER_LOCK:
            return self._slot.loaded

    @_loaded.setter
    def _loaded(self, value: bool) -> None:
        """Update the loaded state on the shared module slot."""
        with _LOADER_LOCK:
            self._slot.loaded = value

    @property
    def _module(self) -> ModuleNamespaceProtocol | None:
        """Cached `ModuleNamespaceProtocol` on the shared module slot."""
        with _LOADER_LOCK:
            return self._slot.module

    @_module.setter
    def _module(self, value: ModuleNamespaceProtocol | None) -> None:
        """Update the cached `ModuleNamespaceProtocol` on the shared module slot."""
        with _LOADER_LOCK:
            self._slot.module = value

    @property
    def is_overridden(self) -> bool:
        """Indicate whether an explicit `override_lazy_module` override is active."""
        with _LOADER_LOCK:
            return self._slot.overridden

    def reset(self) -> None:
        """Reset the cached module state so the next `get_module()` re-imports."""
        with _LOADER_LOCK:
            self._slot.module = None
            self._slot.loaded = False
            self._slot.overridden = False

    def set_override(
        self,
        mock_mod: ModuleNamespaceProtocol | None,
    ) -> tuple[bool, bool, ModuleNamespaceProtocol | None]:
        """Set a temporary module override and return the previous `(loaded, overridden, module)` state.

        Args:
            mock_mod: Replacement module namespace or `None`.

        Returns:
            Tuple of `(previous_loaded, previous_overridden, previous_module)`.

        """
        with _LOADER_LOCK:
            prev = (self._slot.loaded, self._slot.overridden, self._slot.module)
            self._slot.loaded = True
            self._slot.overridden = True
            self._slot.module = mock_mod
            return prev

    def restore_state(self, state: tuple[bool, bool, ModuleNamespaceProtocol | None]) -> None:
        """Restore a previously saved `(loaded, overridden, module)` state.

        Args:
            state: Tuple of `(previous_loaded, previous_overridden, previous_module)`.

        """
        with _LOADER_LOCK:
            self._slot.loaded, self._slot.overridden, self._slot.module = state

    def get_module(self) -> T_co | None:
        """Provide the imported module validated via `self._validator`, or `None` if unavailable.

        Returns:
            Validated `T_co` module instance, or `None` if unavailable or failing protocol validation.

        """
        with _LOADER_LOCK:
            if self._slot.overridden:
                overridden_mod = self._slot.module
                return self._validator(overridden_mod) if overridden_mod is not None else None
            if self._importer is None and self.module_name in sys.modules:
                sys_mod = sys.modules.get(self.module_name)
                if sys_mod is None:
                    return None
                self._slot.module = sys_mod
                self._slot.loaded = True
                return self._validator(sys_mod)
            if not self._slot.loaded:
                importer_fn = self._importer if self._importer is not None else _default_import_module
                try:
                    self._slot.module = importer_fn(self.module_name)
                except (ImportError, AttributeError) as err:
                    logger.debug("Optional module %s not available: %s", self.module_name, err)
                    self._slot.module = None
                self._slot.loaded = True
            raw_mod = self._slot.module
            return self._validator(raw_mod) if raw_mod is not None else None

    @property
    def is_available(self) -> bool:
        """Indicate whether the target module can be imported and satisfies `_validator`."""
        return self.get_module() is not None


class _AllocatedLazyLoader(LazyLoader[TSymbol]):
    """Concrete `LazyLoader` subclass bypassing flyweight `__new__` recursion."""

    def __new__(
        cls,
        module_name: str,
        slot: _ModuleSlot | type[TSymbol] | None = None,
        validator: Callable[[ModuleNamespaceProtocol], TSymbol | None] | None = None,
        importer: ModuleImporterCallable | None = None,
    ) -> _AllocatedLazyLoader[TSymbol]:
        """Allocate and populate a concrete `_AllocatedLazyLoader[TSymbol]` instance.

        Args:
            module_name: Dotted module name string.
            slot: Shared `_ModuleSlot` instance.
            validator: Narrowing callable for loaded module namespaces.
            importer: Optional custom module importer callable.

        Returns:
            Initialized `_AllocatedLazyLoader[TSymbol]` instance.

        """
        assert isinstance(slot, _ModuleSlot)
        assert validator is not None
        inst = super(LazyLoader, cls).__new__(cls)
        inst.module_name = module_name
        inst._slot = slot
        inst._validator = validator
        inst._importer = importer
        return inst

    def __init__(
        self,
        module_name: str,
        slot: _ModuleSlot | type[TSymbol] | None = None,
        validator: Callable[[ModuleNamespaceProtocol], TSymbol | None] | None = None,
        importer: ModuleImporterCallable | None = None,
    ) -> None:
        """No-op initializer preserving fields populated in `__new__`.

        Args:
            module_name: Dotted module name string.
            slot: Shared `_ModuleSlot` or optional protocol argument forwarded from `LazyLoader(...)`.
            validator: Narrowing callable for loaded module namespaces.
            importer: Optional custom module importer callable.

        """
        _ = (module_name, slot, validator, importer)


def reset_all_lazy_loaders() -> None:
    """Reset all registered `LazyLoader` slots in a thread-safe manner."""
    with _LOADER_LOCK:
        for loader in _LOADER_REGISTRY.values():
            loader.reset()


@contextmanager
def override_lazy_module(
    module_name: str,
    mock_mod: ModuleNamespaceProtocol | None,
) -> Generator[None, None, None]:
    """Temporarily override the `LazyLoader` slot for `module_name` within a context block.

    Args:
        module_name: Dotted module name registered in `_LOADER_REGISTRY`.
        mock_mod: Replacement module namespace or `None` to simulate a missing dependency.

    Yields:
        None while the override is active.

    """
    with _LOADER_LOCK:
        loader = LazyLoader(module_name)
        prev_state = loader.set_override(mock_mod)
    try:
        yield
    finally:
        with _LOADER_LOCK:
            loader.restore_state(prev_state)


@overload
def optional_import(module_name: str, protocol: None = None) -> ModuleNamespaceProtocol | None:
    """Import an optional module using `LazyLoader`, returning `None` if unavailable."""


@overload
def optional_import(module_name: str, protocol: type[T_co]) -> T_co | None:
    """Import an optional module and validate `isinstance(mod, protocol)`."""


def optional_import(
    module_name: str,
    protocol: type[T_co] | None = None,
) -> T_co | ModuleNamespaceProtocol | None:
    """Import an optional module using `LazyLoader`, returning `None` if unavailable.

    Args:
        module_name: Dotted module name string to import.
        protocol: Optional `@runtime_checkable` protocol or type to validate against.

    Returns:
        Validated module instance or `None` if unavailable.

    """
    if protocol is None:
        base_loader = LazyLoader(module_name)
        base_mod = base_loader.get_module()
        if base_loader.is_overridden:
            return base_mod
        if module_name in sys.modules:
            return sys.modules.get(module_name)
        return base_mod

    typed_loader = LazyLoader.for_protocol(module_name, protocol)
    typed_mod = typed_loader.get_module()
    if typed_loader.is_overridden:
        return typed_mod
    if module_name in sys.modules:
        sys_mod = sys.modules.get(module_name)
        if sys_mod is None:
            return None
        return sys_mod if isinstance(sys_mod, protocol) else None
    return typed_mod


def import_optional_submodule(module_name: str, submodule_name: str) -> ModuleNamespaceProtocol | None:
    """Import an optional submodule `f'{module_name}.{submodule_name}'` without `getattr`.

    Args:
        module_name: Parent dotted module name (e.g., `'PIL'`, `'etils'`).
        submodule_name: Submodule name (e.g., `'Image'`, `'epath'`).

    Returns:
        Imported submodule `ModuleNamespaceProtocol` or `None` if unavailable.

    """
    parent_loader = LazyLoader(module_name)
    parent = optional_import(module_name)
    if parent is None:
        return None
    ns = vars(parent)
    if submodule_name in ns:
        existing = ns[submodule_name]
        if isinstance(existing, SubmoduleNamespaceProtocol):
            return existing
    if parent_loader.is_overridden:
        return None
    return optional_import(f"{module_name}.{submodule_name}")


def import_optional_symbol(
    module_name: str,
    symbol_name: str,
    protocol: type[TSymbol],
) -> TSymbol | None:
    """Import an optional symbol from `module_name` and verify `isinstance(val, protocol)` without `getattr`.

    Resolves `symbol_name` from `vars(module)` first, then checks canonical lazy re-export
    submodules (such as `transformers.models.auto.modeling_auto`) or direct submodule paths
    via `importlib.import_module`, verifying runtime structural conformance via `isinstance`.

    Args:
        module_name: Dotted module name string to import.
        symbol_name: Exported class, function, or factory symbol name.
        protocol: `@runtime_checkable` protocol or runtime class that the symbol must satisfy.

    Returns:
        Verified `TSymbol` instance, or `None` if unavailable or failing `isinstance(..., protocol)`.

    """
    loader = LazyLoader(module_name)
    mod = optional_import(module_name)
    if mod is None:
        return None
    mod_ns = vars(mod)
    if symbol_name in mod_ns:
        candidate = mod_ns[symbol_name]
        return candidate if isinstance(candidate, protocol) else None
    if loader.is_overridden:
        return None

    reexport_mod_name = _LAZY_REEXPORT_SUBMODULES.get((module_name, symbol_name))
    if reexport_mod_name is not None:
        reexport_mod = optional_import(reexport_mod_name)
        if reexport_mod is not None and symbol_name in vars(reexport_mod):
            reexport_val = vars(reexport_mod)[symbol_name]
            return reexport_val if isinstance(reexport_val, protocol) else None

    sub_mod = optional_import(f"{module_name}.{symbol_name}")
    if sub_mod is not None and isinstance(sub_mod, protocol):
        return sub_mod
    return None


_JAX_LOADER: LazyLoader[ModuleNamespaceProtocol] = LazyLoader("jax")
_JNP_LOADER: LazyLoader[ModuleNamespaceProtocol] = LazyLoader("jax.numpy")
_FLAX_NNX_LOADER: LazyLoader[ModuleNamespaceProtocol] = LazyLoader("flax.nnx")
_TENSORFLOW_LOADER: LazyLoader[ModuleNamespaceProtocol] = LazyLoader("tensorflow")
_KERAS_LOADER: LazyLoader[ModuleNamespaceProtocol] = LazyLoader("keras")
_TORCH_LOADER: LazyLoader[ModuleNamespaceProtocol] = LazyLoader("torch")
_MLX_LOADER: LazyLoader[ModuleNamespaceProtocol] = LazyLoader("mlx")
_DUCKDB_LOADER: LazyLoader[ModuleNamespaceProtocol] = LazyLoader("duckdb")
_TRANSFORMERS_LOADER: LazyLoader[ModuleNamespaceProtocol] = LazyLoader("transformers")
_SAFETENSORS_LOADER: LazyLoader[ModuleNamespaceProtocol] = LazyLoader("safetensors")
_MAXTEXT_GEMMA4_LOADER: LazyLoader[ModuleNamespaceProtocol] = LazyLoader("maxtext.models.gemma4")


def get_jax() -> ModuleNamespaceProtocol | None:
    """Provide the `jax` module namespace if installed.

    Returns:
        The `jax` module namespace or `None`.

    """
    return _JAX_LOADER.get_module()


def get_jnp() -> ModuleNamespaceProtocol | None:
    """Provide the `jax.numpy` module namespace if installed.

    Returns:
        The `jax.numpy` module namespace or `None`.

    """
    return _JNP_LOADER.get_module()


def get_flax_nnx() -> ModuleNamespaceProtocol | None:
    """Provide the `flax.nnx` module namespace if installed.

    Returns:
        The `flax.nnx` module namespace or `None`.

    """
    return _FLAX_NNX_LOADER.get_module()


def get_tensorflow() -> ModuleNamespaceProtocol | None:
    """Provide the `tensorflow` module namespace if installed.

    Returns:
        The `tensorflow` module namespace or `None`.

    """
    return _TENSORFLOW_LOADER.get_module()


def get_keras() -> ModuleNamespaceProtocol | None:
    """Provide the `keras` module namespace if installed.

    Returns:
        The `keras` module namespace or `None`.

    """
    return _KERAS_LOADER.get_module()


def get_torch() -> ModuleNamespaceProtocol | None:
    """Provide the `torch` module namespace if installed.

    Returns:
        The `torch` module namespace or `None`.

    """
    return _TORCH_LOADER.get_module()


def get_mlx() -> ModuleNamespaceProtocol | None:
    """Provide the `mlx` module namespace if installed.

    Returns:
        The `mlx` module namespace or `None`.

    """
    return _MLX_LOADER.get_module()


def get_duckdb() -> ModuleNamespaceProtocol | None:
    """Provide the `duckdb` module namespace if installed.

    Returns:
        The `duckdb` module namespace or `None`.

    """
    return _DUCKDB_LOADER.get_module()


def get_transformers() -> ModuleNamespaceProtocol | None:
    """Provide the `transformers` module namespace if installed.

    Returns:
        The `transformers` module namespace or `None`.

    """
    return _TRANSFORMERS_LOADER.get_module()


def get_safetensors() -> ModuleNamespaceProtocol | None:
    """Provide the `safetensors` module namespace if installed.

    Returns:
        The `safetensors` module namespace or `None`.

    """
    return _SAFETENSORS_LOADER.get_module()


def get_maxtext_gemma4() -> ModuleNamespaceProtocol | None:
    """Provide the `maxtext.models.gemma4` module namespace if installed.

    Returns:
        The `maxtext.models.gemma4` module namespace or `None`.

    """
    return _MAXTEXT_GEMMA4_LOADER.get_module()


__all__ = [
    "OPTIONAL_IMPORT_ERRORS",
    "LazyLoader",
    "ModuleImporterCallable",
    "ModuleNamespaceProtocol",
    "SubmoduleNamespaceProtocol",
    "catch_optional_imports",
    "get_duckdb",
    "get_flax_nnx",
    "get_jax",
    "get_jnp",
    "get_keras",
    "get_maxtext_gemma4",
    "get_mlx",
    "get_safetensors",
    "get_tensorflow",
    "get_torch",
    "get_transformers",
    "import_optional_submodule",
    "import_optional_symbol",
    "optional_import",
    "override_lazy_module",
    "reset_all_lazy_loaders",
]

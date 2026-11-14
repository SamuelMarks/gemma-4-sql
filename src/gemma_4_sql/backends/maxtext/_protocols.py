"""Shared runtime-checkable module protocols for the MaxText backend."""

from __future__ import annotations

import types
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol, Self, runtime_checkable

from gemma_4_sql.exceptions import StatusValue
from gemma_4_sql.type_hints import (
    DTypeProtocol,
    JaxArrayProtocol,
    JSONValue,
    PyTreeNode,
)


@runtime_checkable
class JaxDistributedNamespaceProtocol(Protocol):
    """Protocol for `jax.distributed` exposing `initialize(**kwargs)`."""

    def initialize(self, **kwargs: str | int | None) -> None:
        """Initialize the JAX multi-host distributed runtime.

        Args:
            **kwargs: Optional `coordinator_address`, `num_processes`, or `process_id` arguments.

        """
        ...


@runtime_checkable
class JaxDistributedHolderProtocol(Protocol):
    """Protocol for a `jax` module exposing `.distributed`."""

    @property
    def distributed(self) -> JaxDistributedNamespaceProtocol:
        """Provide `jax.distributed` namespace."""
        ...


@runtime_checkable
class JaxRandomPRNGNamespaceProtocol(Protocol):
    """Protocol for `jax.random` exposing `PRNGKey`."""

    def PRNGKey(self, seed: int, /) -> JaxArrayProtocol:
        """Create a deterministic JAX PRNG key array.

        Args:
            seed: Integer PRNG seed.

        Returns:
            PRNG key `JaxArrayProtocol`.

        """
        ...


@runtime_checkable
class JaxPRNGHolderProtocol(Protocol):
    """Protocol for a `jax` module exposing `.random.PRNGKey(...)`."""

    @property
    def random(self) -> JaxRandomPRNGNamespaceProtocol:
        """Provide `jax.random` namespace."""
        ...


@runtime_checkable
class JaxDistributedAndPRNGProtocol(JaxDistributedHolderProtocol, JaxPRNGHolderProtocol, Protocol):
    """Protocol for a `jax` module exposing both `.distributed` and `.random`."""


@runtime_checkable
class JnpZerosInt32Protocol(Protocol):
    """Protocol for `jax.numpy` exposing `int32` and `zeros`."""

    @property
    def int32(self) -> DTypeProtocol:
        """Provide `jnp.int32` dtype."""
        ...

    def zeros(
        self,
        shape: tuple[int, ...],
        *,
        dtype: DTypeProtocol | str | None = None,
    ) -> JaxArrayProtocol:
        """Construct a zero-filled `JaxArrayProtocol`.

        Args:
            shape: Output array dimensions.
            dtype: Optional target dtype.

        Returns:
            Zero-initialized `JaxArrayProtocol`.

        """
        ...


@runtime_checkable
class JnpZerosAndArrayProtocol(JnpZerosInt32Protocol, Protocol):
    """Protocol for `jax.numpy` exposing `int32`, `zeros`, and `array`."""

    def array(
        self,
        obj: Sequence[Sequence[int]] | Sequence[int] | float,
        *,
        dtype: DTypeProtocol | str | None = None,
    ) -> JaxArrayProtocol:
        """Construct a `JaxArrayProtocol` from nested sequences or scalars.

        Args:
            obj: Input sequence or scalar.
            dtype: Optional target dtype.

        Returns:
            Constructed `JaxArrayProtocol`.

        """
        ...


@runtime_checkable
class OrbaxCheckpointerInstanceProtocol(Protocol):
    """Structural protocol for an `orbax.checkpoint.PyTreeCheckpointer` instance."""


@runtime_checkable
class OrbaxOptionsInstanceProtocol(Protocol):
    """Structural protocol for an `orbax.checkpoint.CheckpointManagerOptions` instance."""


@runtime_checkable
class OrbaxManagerProtocol(Protocol):
    """Protocol for an `orbax.checkpoint.CheckpointManager` context manager."""

    def __enter__(self) -> Self:
        """Enter checkpoint manager context.

        Returns:
            Active checkpoint manager.

        """
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
        /,
    ) -> bool | None:
        """Exit checkpoint manager context.

        Args:
            exc_type: Exception class if raised.
            exc_val: Exception instance if raised.
            exc_tb: Traceback object if raised.

        Returns:
            Optional suppression flag.

        """
        ...

    def latest_step(self) -> int | None:
        """Return latest persisted step integer or `None`.

        Returns:
            Latest step index or `None`.

        """
        ...

    def restore(self, step: int, /) -> Mapping[str, PyTreeNode]:
        """Restore checkpoint PyTree at `step`.

        Args:
            step: Checkpoint step integer.

        Returns:
            Restored parameter PyTree dictionary.

        """
        ...

    def save[ItemT](
        self,
        step: int,
        items: StatusValue | JSONValue | Mapping[str, ItemT],
        /,
    ) -> bool | None:
        """Persist `items` at `step`.

        Args:
            step: Checkpoint step integer.
            items: Parameter or training state PyTree.

        Returns:
            Optional boolean save status or `None`.

        """
        ...


@runtime_checkable
class OrbaxCheckpointModuleProtocol(Protocol):
    """Protocol for `orbax.checkpoint` exposing `CheckpointManager`, `PyTreeCheckpointer`, and options."""

    def PyTreeCheckpointer(self) -> OrbaxCheckpointerInstanceProtocol:
        """Construct a `PyTreeCheckpointer`.

        Returns:
            Initialized `OrbaxCheckpointerInstanceProtocol`.

        """
        ...

    def CheckpointManagerOptions(
        self,
        *,
        max_to_keep: int = 1,
        create: bool = True,
    ) -> OrbaxOptionsInstanceProtocol:
        """Construct `CheckpointManagerOptions`.

        Args:
            max_to_keep: Maximum retained checkpoint steps.
            create: Whether to create the checkpoint directory if absent.

        Returns:
            Initialized `OrbaxOptionsInstanceProtocol`.

        """
        ...

    def CheckpointManager(
        self,
        directory: Path | str,
        checkpointers: OrbaxCheckpointerInstanceProtocol,
        options: OrbaxOptionsInstanceProtocol | None = None,
        /,
    ) -> OrbaxManagerProtocol:
        """Construct a `CheckpointManager`.

        Args:
            directory: Target checkpoint directory.
            checkpointers: Underlying PyTree checkpointer.
            options: Optional retention and creation options.

        Returns:
            Initialized `OrbaxManagerProtocol`.

        """
        ...


__all__ = [
    "JaxDistributedAndPRNGProtocol",
    "JaxDistributedHolderProtocol",
    "JaxDistributedNamespaceProtocol",
    "JaxPRNGHolderProtocol",
    "JaxRandomPRNGNamespaceProtocol",
    "JnpZerosAndArrayProtocol",
    "JnpZerosInt32Protocol",
    "OrbaxCheckpointModuleProtocol",
    "OrbaxCheckpointerInstanceProtocol",
    "OrbaxManagerProtocol",
    "OrbaxOptionsInstanceProtocol",
]

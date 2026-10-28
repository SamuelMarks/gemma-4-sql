"""Runtime keyword-argument extractors and synchronous coroutine execution helpers."""

from __future__ import annotations

from collections.abc import Callable, Coroutine, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Protocol, TypeVar, Union, runtime_checkable

type KwargScalar = str | int | float | bool | Path | None
type KwargValue = Union[
    KwargScalar,
    Sequence["KwargValue"],
    Mapping[str, "KwargValue"],
]


@runtime_checkable
class KwargMappingProtocol(Protocol):
    """Protocol satisfied by all string-keyed mappings and TypedDicts."""

    def __iter__(self) -> Iterator[str]:
        """Iterate over the string keys of the mapping."""
        ...

    def __len__(self) -> int:
        """Return the number of entries in the mapping."""
        ...


type KwargMapping = Mapping[str, KwargValue] | KwargMappingProtocol


def extract_int_kwarg(kwargs: KwargMapping, key: str, default: int) -> int:
    """Extract an integer value from a keyword-argument mapping without string roundtrip coercion.

    Args:
        kwargs: Mapping of keyword arguments.
        key: Key to look up.
        default: Default integer returned when key is absent or None.

    Returns:
        Coerced integer value.

    """
    val = kwargs.get(key) if isinstance(kwargs, Mapping) else None
    if val is None:
        return default
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, (int, float, str)):
        return int(val)
    return default


def extract_float_kwarg(kwargs: KwargMapping, key: str, default: float) -> float:
    """Extract a float value from a keyword-argument mapping without string roundtrip coercion.

    Args:
        kwargs: Mapping of keyword arguments.
        key: Key to look up.
        default: Default float returned when key is absent or None.

    Returns:
        Coerced float value.

    """
    val = kwargs.get(key) if isinstance(kwargs, Mapping) else None
    if val is None:
        return default
    if isinstance(val, bool):
        return float(val)
    if isinstance(val, (int, float, str)):
        return float(val)
    return default


def extract_str_kwarg(kwargs: KwargMapping, key: str, default: str | None = None) -> str | None:
    """Extract an optional string value from a keyword-argument mapping.

    Args:
        kwargs: Mapping of keyword arguments.
        key: Key to look up.
        default: Default string or None returned when key is absent or None.

    Returns:
        Extracted string or default.

    """
    val = kwargs.get(key) if isinstance(kwargs, Mapping) else None
    if val is None:
        return default
    return str(val)


def extract_bool_kwarg(kwargs: KwargMapping, key: str, default: bool = False) -> bool:
    """Extract a boolean value from a keyword-argument mapping.

    Args:
        kwargs: Mapping of keyword arguments.
        key: Key to look up.
        default: Default boolean returned when key is absent or None.

    Returns:
        Extracted boolean value.

    """
    val = kwargs.get(key) if isinstance(kwargs, Mapping) else None
    if val is None:
        return default
    if isinstance(val, str):
        return val.strip().lower() in ("1", "true", "yes", "on")
    return bool(val)


TResult = TypeVar("TResult")


def run_coroutine_sync(coro_factory: Callable[[], Coroutine[None, None, TResult]]) -> TResult:
    """Execute an async coroutine factory synchronously, safely offloading to a worker thread if an event loop is already running.

    Args:
        coro_factory: Zero-argument callable returning the coroutine to await.

    Returns:
        The resolved return value of the coroutine.

    """
    import asyncio
    import concurrent.futures

    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    if running_loop is not None and running_loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(lambda: asyncio.run(coro_factory())).result()
    return asyncio.run(coro_factory())


__all__ = [
    "KwargScalar",
    "KwargValue",
    "TResult",
    "extract_bool_kwarg",
    "extract_float_kwarg",
    "extract_int_kwarg",
    "extract_str_kwarg",
    "run_coroutine_sync",
]

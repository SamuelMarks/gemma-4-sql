"""JSON primitive, recursive value, mapping, and status type definitions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Union

from gemma_4_sql._types_extractors import (
    KwargScalar,
    KwargValue,
    StatusScalar,
    TResult,
    extract_bool_kwarg,
    extract_float_kwarg,
    extract_int_kwarg,
    extract_status_kwargs,
    extract_str_kwarg,
    run_coroutine_sync,
    shutdown_sync_coro_executor,
)

JSONPrimitive = Union[str, int, float, bool, None]
JSONValue = Union[JSONPrimitive, Sequence["JSONValue"], Mapping[str, "JSONValue"]]
JSONMapping = Mapping[str, JSONValue]
JSONDict = dict[str, JSONValue]

__all__ = [
    "JSONDict",
    "JSONMapping",
    "JSONPrimitive",
    "JSONValue",
    "KwargScalar",
    "KwargValue",
    "StatusScalar",
    "TResult",
    "extract_bool_kwarg",
    "extract_float_kwarg",
    "extract_int_kwarg",
    "extract_status_kwargs",
    "extract_str_kwarg",
    "run_coroutine_sync",
    "shutdown_sync_coro_executor",
]

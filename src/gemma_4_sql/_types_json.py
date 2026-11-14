"""JSON primitive, recursive value, and mapping type definitions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Union

JSONPrimitive = Union[str, int, float, bool, None]
JSONValue = Union[JSONPrimitive, Sequence["JSONValue"], Mapping[str, "JSONValue"]]
JSONMapping = Mapping[str, JSONValue]
JSONDict = dict[str, JSONValue]

__all__ = [
    "JSONDict",
    "JSONMapping",
    "JSONPrimitive",
    "JSONValue",
]

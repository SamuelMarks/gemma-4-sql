"""Custom type hints for gemma-4-sql."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeAlias

JSONPrimitive: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONPrimitive | Sequence["JSONValue"] | Mapping[str, "JSONValue"]
JSONDict: TypeAlias = dict[str, JSONValue]

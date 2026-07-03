"""Custom type hints for gemma-4-sql."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

JSONPrimitive = str | int | float | bool | None
JSONValue = JSONPrimitive | Sequence["JSONValue"] | Mapping[str, "JSONValue"]
JSONDict = dict[str, Any]

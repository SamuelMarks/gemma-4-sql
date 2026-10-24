"""MaxText backend adapter."""

from __future__ import annotations

from gemma_4_sql.backends.base_adapter import ModuleDelegatedBackend

from . import benchmark, dpo, etl, export, inference, logging, peft, quantize, serve, train


class MaxTextBackend(ModuleDelegatedBackend):
    """MaxText implementation of BackendProtocol."""


ADAPTER_CLASS = MaxTextBackend

__all__ = [
    "ADAPTER_CLASS",
    "MaxTextBackend",
    "benchmark",
    "dpo",
    "etl",
    "export",
    "inference",
    "logging",
    "peft",
    "quantize",
    "serve",
    "train",
]

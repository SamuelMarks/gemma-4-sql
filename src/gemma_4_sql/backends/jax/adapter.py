"""JAX backend adapter."""

from __future__ import annotations

from gemma_4_sql.backends.base_adapter import ModuleDelegatedBackend

from . import benchmark, dpo, etl, export, inference, logging, peft, quantize, serve, train


class JaxBackend(ModuleDelegatedBackend):
    """JAX implementation of BackendProtocol."""


ADAPTER_CLASS = JaxBackend

__all__ = [
    "ADAPTER_CLASS",
    "JaxBackend",
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

"""MLX backend adapter."""

from __future__ import annotations

from gemma_4_sql.backends.base_adapter import ModuleDelegatedBackend

from . import benchmark, dpo, etl, export, inference, logging, peft, quantize, serve, train


class MLXBackend(ModuleDelegatedBackend):
    """MLX implementation of BackendProtocol."""


ADAPTER_CLASS = MLXBackend

__all__ = [
    "ADAPTER_CLASS",
    "MLXBackend",
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

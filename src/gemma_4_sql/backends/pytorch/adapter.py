"""PyTorch backend adapter."""

from __future__ import annotations

from gemma_4_sql.backends.base_adapter import ModuleDelegatedBackend

from . import benchmark, dpo, etl, export, inference, logging, peft, quantize, serve, train


class PyTorchBackend(ModuleDelegatedBackend):
    """PyTorch implementation of BackendProtocol."""


ADAPTER_CLASS = PyTorchBackend

__all__ = [
    "ADAPTER_CLASS",
    "PyTorchBackend",
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

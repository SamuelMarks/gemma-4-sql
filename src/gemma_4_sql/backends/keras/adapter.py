"""Keras backend adapter."""

from __future__ import annotations

from gemma_4_sql.backends.base_adapter import ModuleDelegatedBackend

from . import benchmark, dpo, etl, export, inference, logging, peft, quantize, serve, train


class KerasBackend(ModuleDelegatedBackend):
    """Keras implementation of BackendProtocol."""


ADAPTER_CLASS = KerasBackend

__all__ = [
    "ADAPTER_CLASS",
    "KerasBackend",
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

"""PyTorch backend adapter."""

from __future__ import annotations

from gemma_4_sql.backends.base_adapter import ModuleDelegatedBackend

from . import benchmark, dpo, etl, export, inference, logging, peft, quantize, serve, train


class PyTorchBackend(ModuleDelegatedBackend):
    """PyTorch implementation of BackendProtocol."""

    train_module = train
    dpo_module = dpo
    etl_module = etl
    export_module = export
    logging_module = logging
    peft_module = peft
    quantize_module = quantize
    inference_module = inference
    serve_module = serve
    benchmark_module = benchmark
    trainer_name = "pytorch_trainer"


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

"""MLX backend adapter."""

from __future__ import annotations

from gemma_4_sql.backends.base_adapter import ModuleDelegatedBackend

from . import benchmark, dpo, etl, export, inference, logging, peft, quantize, serve, train


class MLXBackend(ModuleDelegatedBackend):
    """MLX implementation of BackendProtocol."""

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
    trainer_name = "mlx_trainer"


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

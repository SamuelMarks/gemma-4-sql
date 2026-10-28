"""JAX backend adapter."""

from __future__ import annotations

from gemma_4_sql.backends.base_adapter import ModuleDelegatedBackend

from . import benchmark, dpo, etl, export, inference, logging, peft, quantize, serve, train


class JaxBackend(ModuleDelegatedBackend):
    """JAX implementation of BackendProtocol."""

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
    trainer_name = "jax_trainer"


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

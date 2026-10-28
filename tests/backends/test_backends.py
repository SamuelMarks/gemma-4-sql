"""Tests verifying public package exports and signature parity across all 5 backends."""

from __future__ import annotations

import inspect

import gemma_4_sql.backends.jax as jax_pkg
import gemma_4_sql.backends.keras as keras_pkg
import gemma_4_sql.backends.maxtext as maxtext_pkg
import gemma_4_sql.backends.mlx as mlx_pkg
import gemma_4_sql.backends.pytorch as pytorch_pkg
from gemma_4_sql.sdk.protocols import BackendProtocol
from gemma_4_sql.sdk.registry import get_backend


def test_backend_apis_exist() -> None:
    """Verify all 5 backend packages export the complete 11-function public API and all 10 submodules."""
    packages = (
        (
            "jax",
            jax_pkg,
            (
                jax_pkg.apply_lora,
                jax_pkg.benchmark_model,
                jax_pkg.build_dataloader,
                jax_pkg.export_model,
                jax_pkg.generate_sql,
                jax_pkg.get_trainer,
                jax_pkg.log_metrics,
                jax_pkg.quantize_model,
                jax_pkg.run_dpo,
                jax_pkg.serve_model,
                jax_pkg.train_model,
            ),
            (
                jax_pkg.benchmark,
                jax_pkg.dpo,
                jax_pkg.etl,
                jax_pkg.export,
                jax_pkg.inference,
                jax_pkg.logging,
                jax_pkg.peft,
                jax_pkg.quantize,
                jax_pkg.serve,
                jax_pkg.train,
            ),
        ),
        (
            "keras",
            keras_pkg,
            (
                keras_pkg.apply_lora,
                keras_pkg.benchmark_model,
                keras_pkg.build_dataloader,
                keras_pkg.export_model,
                keras_pkg.generate_sql,
                keras_pkg.get_trainer,
                keras_pkg.log_metrics,
                keras_pkg.quantize_model,
                keras_pkg.run_dpo,
                keras_pkg.serve_model,
                keras_pkg.train_model,
            ),
            (
                keras_pkg.benchmark,
                keras_pkg.dpo,
                keras_pkg.etl,
                keras_pkg.export,
                keras_pkg.inference,
                keras_pkg.logging,
                keras_pkg.peft,
                keras_pkg.quantize,
                keras_pkg.serve,
                keras_pkg.train,
            ),
        ),
        (
            "maxtext",
            maxtext_pkg,
            (
                maxtext_pkg.apply_lora,
                maxtext_pkg.benchmark_model,
                maxtext_pkg.build_dataloader,
                maxtext_pkg.export_model,
                maxtext_pkg.generate_sql,
                maxtext_pkg.get_trainer,
                maxtext_pkg.log_metrics,
                maxtext_pkg.quantize_model,
                maxtext_pkg.run_dpo,
                maxtext_pkg.serve_model,
                maxtext_pkg.train_model,
            ),
            (
                maxtext_pkg.benchmark,
                maxtext_pkg.dpo,
                maxtext_pkg.etl,
                maxtext_pkg.export,
                maxtext_pkg.inference,
                maxtext_pkg.logging,
                maxtext_pkg.peft,
                maxtext_pkg.quantize,
                maxtext_pkg.serve,
                maxtext_pkg.train,
            ),
        ),
        (
            "mlx",
            mlx_pkg,
            (
                mlx_pkg.apply_lora,
                mlx_pkg.benchmark_model,
                mlx_pkg.build_dataloader,
                mlx_pkg.export_model,
                mlx_pkg.generate_sql,
                mlx_pkg.get_trainer,
                mlx_pkg.log_metrics,
                mlx_pkg.quantize_model,
                mlx_pkg.run_dpo,
                mlx_pkg.serve_model,
                mlx_pkg.train_model,
            ),
            (
                mlx_pkg.benchmark,
                mlx_pkg.dpo,
                mlx_pkg.etl,
                mlx_pkg.export,
                mlx_pkg.inference,
                mlx_pkg.logging,
                mlx_pkg.peft,
                mlx_pkg.quantize,
                mlx_pkg.serve,
                mlx_pkg.train,
            ),
        ),
        (
            "pytorch",
            pytorch_pkg,
            (
                pytorch_pkg.apply_lora,
                pytorch_pkg.benchmark_model,
                pytorch_pkg.build_dataloader,
                pytorch_pkg.export_model,
                pytorch_pkg.generate_sql,
                pytorch_pkg.get_trainer,
                pytorch_pkg.log_metrics,
                pytorch_pkg.quantize_model,
                pytorch_pkg.run_dpo,
                pytorch_pkg.serve_model,
                pytorch_pkg.train_model,
            ),
            (
                pytorch_pkg.benchmark,
                pytorch_pkg.dpo,
                pytorch_pkg.etl,
                pytorch_pkg.export,
                pytorch_pkg.inference,
                pytorch_pkg.logging,
                pytorch_pkg.peft,
                pytorch_pkg.quantize,
                pytorch_pkg.serve,
                pytorch_pkg.train,
            ),
        ),
    )
    for name, pkg, exported_fns, submodules in packages:
        for fn in exported_fns:
            assert callable(fn)
            sig = inspect.signature(fn)
            assert sig.return_annotation is not inspect.Signature.empty
        for submod in submodules:
            assert submod is not None
        assert pkg.get_trainer() == f"{name}_trainer"

        resolved_backend = get_backend(name)
        assert isinstance(resolved_backend, BackendProtocol)
        assert resolved_backend.get_trainer() == f"{name}_trainer"

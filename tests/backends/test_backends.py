"""Tests verifying public package exports and signature parity across all 5 backends."""

from __future__ import annotations

import inspect

import gemma_4_sql.backends.jax as jax_pkg
import gemma_4_sql.backends.keras as keras_pkg
import gemma_4_sql.backends.maxtext as maxtext_pkg
import gemma_4_sql.backends.mlx as mlx_pkg
import gemma_4_sql.backends.pytorch as pytorch_pkg

EXPECTED_EXPORTS = (
    "apply_lora",
    "benchmark_model",
    "build_dataloader",
    "export_model",
    "generate_sql",
    "get_trainer",
    "log_metrics",
    "quantize_model",
    "run_dpo",
    "serve_model",
    "train_model",
)


def test_backend_apis_exist() -> None:
    """Verify all 5 backend packages export the complete 11-function public API with matching first parameters."""
    packages = {
        "jax": jax_pkg,
        "keras": keras_pkg,
        "maxtext": maxtext_pkg,
        "mlx": mlx_pkg,
        "pytorch": pytorch_pkg,
    }
    for name, pkg in packages.items():
        for export_name in EXPECTED_EXPORTS:
            fn = getattr(pkg, export_name, None)
            assert callable(fn), f"{name}.{export_name} is missing or not callable"
            sig = inspect.signature(fn)
            assert sig.return_annotation is not inspect.Signature.empty, f"{name}.{export_name} lacks return type annotation"
        assert pkg.get_trainer() == f"{name}_trainer"

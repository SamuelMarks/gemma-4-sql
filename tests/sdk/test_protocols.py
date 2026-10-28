"""Tests for backend protocols and runtime conformance across all registered backends."""

from __future__ import annotations

import inspect

import pytest

from gemma_4_sql.sdk.protocols import BackendProtocol, InferenceProtocol, TrainingProtocol
from gemma_4_sql.sdk.registry import get_backend


@pytest.mark.parametrize("backend_name", ["jax", "keras", "maxtext", "pytorch", "pytorch_hf", "pytorch_native", "mlx"])
def test_registered_backends_conform_to_protocol(backend_name: str) -> None:
    """Verify that every registered backend satisfies runtime_checkable BackendProtocol."""
    backend = get_backend(backend_name)

    assert isinstance(backend, TrainingProtocol)
    assert isinstance(backend, InferenceProtocol)
    assert isinstance(backend, BackendProtocol)

    methods = (
        backend.train_model,
        backend.run_dpo,
        backend.build_dataloader,
        backend.export_model,
        backend.log_metrics,
        backend.apply_lora,
        backend.quantize_model,
        backend.get_trainer,
        backend.generate_sql,
        backend.serve_model,
        backend.benchmark_model,
    )
    for method in methods:
        assert callable(method)
        sig = inspect.signature(method)
        assert "kwargs" in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        assert sig.return_annotation != inspect.Signature.empty

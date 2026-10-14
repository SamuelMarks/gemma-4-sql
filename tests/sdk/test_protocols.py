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

    required_methods = [m for m in dir(BackendProtocol) if not m.startswith("_") and not m.isupper()]
    for method_name in required_methods:
        assert hasattr(backend, method_name), f"Backend '{backend_name}' is missing method '{method_name}'"
        method = getattr(backend, method_name)
        assert callable(method), f"Backend '{backend_name}' attribute '{method_name}' is not callable"
        sig = inspect.signature(method)
        assert "kwargs" in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        assert sig.return_annotation != inspect.Signature.empty

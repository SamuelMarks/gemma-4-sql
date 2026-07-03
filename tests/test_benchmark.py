"""Unified tests for Benchmark across all backends."""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock

import pytest

BACKENDS = ["jax", "keras", "maxtext", "mlx", "pytorch"]


class UniversalMock(MagicMock):
    """Mock."""

    def __call__(self, *_args: object, **kwargs: object) -> object:
        """Execute function."""
        if "num_records" in kwargs:
            return self
        return UniversalMock()

    def numpy(self) -> float:
        """Execute function."""
        return 0.0

    def numel(self) -> int:
        """Execute function."""
        return 1000

    def to(self, *_a: object, **_k: object) -> object:
        """Execute function."""
        return self

    def eval(self) -> object:
        """Execute function."""
        return self


@pytest.fixture
def mock_benchmark_backend(request: object, monkeypatch: object) -> object:
    """Mock."""
    backend = request.param
    module = importlib.import_module(f"gemma_4_sql.backends.{backend}.benchmark")
    um = UniversalMock()
    if backend == "jax":
        monkeypatch.setattr(module, "jax", um)
        monkeypatch.setattr(module, "jnp", um)
        monkeypatch.setattr(module, "nnx", um)
        monkeypatch.setattr(module, "Gemma4ForCausalLM", um)
        monkeypatch.setattr(module, "Gemma4Config", um)
    elif backend == "keras":
        monkeypatch.setattr(module, "tf", um)
        monkeypatch.setattr(module, "keras", um)
    elif backend == "maxtext":
        monkeypatch.setattr(module, "jax", um)
        monkeypatch.setattr(module, "jnp", um)
        monkeypatch.setattr(module, "Gemma4Model", um)
    elif backend == "mlx":
        monkeypatch.setattr(module, "mlx", um)
        monkeypatch.setattr(module, "AutoModelForCausalLM", um)
    elif backend == "pytorch":
        monkeypatch.setattr(module, "torch", um)
        monkeypatch.setattr(module, "AutoModelForCausalLM", um)
    return (backend, module)


@pytest.mark.parametrize("mock_benchmark_backend", BACKENDS, indirect=True)
def test_benchmark_model_real(mock_benchmark_backend: object) -> None:
    """Test."""
    (_backend, _module) = mock_benchmark_backend

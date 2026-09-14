"""Tests for backend protocols and conformance across all registered backends."""

from __future__ import annotations

import pytest

from gemma_4_sql.sdk.protocols import BackendProtocol
from gemma_4_sql.sdk.registry import get_backend
from gemma_4_sql.type_hints import DPOConfig, ETLConfig, TrainingConfig


def test_mock_backend_protocol() -> None:
    """Verify that a compliant mock backend implements all protocol methods."""

    class MockBackend:
        """Mock backend adhering to BackendProtocol."""

        def train_model(self, config: TrainingConfig, **kwargs: object) -> dict:
            """Mock train_model."""
            return {"status": "trained"}

        def run_dpo(self, config: DPOConfig, **kwargs: object) -> dict:
            """Mock run_dpo."""
            return {"status": "dpo"}

        def build_dataloader(self, config: ETLConfig, **kwargs: object) -> dict:
            """Mock build_dataloader."""
            return {"loader": None}

        def export_model(self, model_name: str, export_path: str, **kwargs: object) -> dict:
            """Mock export_model."""
            return {"status": "exported"}

        def log_metrics(self, metrics: dict[str, float], step: int, log_dir: str = "logs", **kwargs: object) -> dict:
            """Mock log_metrics."""
            return {"status": "logged"}

        def apply_lora(
            self,
            model_name: str,
            target_modules: list[str],
            lora_r: int = 8,
            lora_alpha: int = 16,
            lora_dropout: float = 0.05,
            **kwargs: object,
        ) -> dict:
            """Mock apply_lora."""
            return {"status": "lora"}

        def quantize_model(self, model_name: str, method: str = "int8", **kwargs: object) -> dict:
            """Mock quantize_model."""
            return {"status": "quantized"}

        def generate_sql(self, model_name: str, prompt: str, beam_width: int = 3, max_length: int = 50, **kwargs: object) -> dict:
            """Mock generate_sql."""
            return {"sql": "SELECT 1"}

        def serve_model(self, model_name: str, port: int = 8000, max_batch_size: int = 32, **kwargs: object) -> dict:
            """Mock serve_model."""
            return {"status": "served"}

        def benchmark_model(self, model_name: str, hardware: str = "tpu-v5p", batch_size: int = 32, **kwargs: object) -> dict:
            """Mock benchmark_model."""
            return {"status": "benchmarked"}

    b = MockBackend()
    assert b.train_model(TrainingConfig())["status"] == "trained"


@pytest.mark.parametrize("backend_name", ["jax", "keras", "maxtext", "pytorch", "pytorch_hf", "pytorch_native", "mlx"])
def test_registered_backends_conform_to_protocol(backend_name: str) -> None:
    """Verify that every registered backend implements all BackendProtocol methods.

    Args:
        backend_name: Name of registered backend entry-point.
    """
    backend = get_backend(backend_name)
    required_methods = [m for m in dir(BackendProtocol) if not m.startswith("_") and not m.isupper()]
    for method_name in required_methods:
        assert hasattr(backend, method_name), f"Backend '{backend_name}' is missing method '{method_name}'"
        assert callable(getattr(backend, method_name)), f"Backend '{backend_name}' attribute '{method_name}' is not callable"

"""Unit and behavioral tests for backend pytorch."""

from gemma_4_sql.backends.pytorch import __all__, get_trainer
from tests.conftest import requires_torch

pytestmark = requires_torch


def test_get_trainer():
    """Execute and verify `test_get_trainer` behavior and invariants."""
    assert get_trainer() == "pytorch_trainer"


def test_all_exports():
    """Execute and verify `test_all_exports` behavior and invariants."""
    expected = [
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
    ]
    assert __all__ == expected

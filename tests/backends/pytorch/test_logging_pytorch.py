"""Tests for PyTorch logging."""

from gemma_4_sql.backends.pytorch.logging import log_metrics


def test_log_metrics() -> None:
    """Test PyTorch logging."""
    metrics = {"loss": 0.5, "acc": 0.9}
    res = log_metrics(metrics, step=10)
    assert res["backend"] == "pytorch"

"""Tests for MaxText logging."""

from gemma_4_sql.backends.maxtext.logging import log_metrics


def test_log_metrics() -> None:
    """Test MaxText logging."""
    metrics = {"loss": 0.5, "acc": 0.9}
    res = log_metrics(metrics, step=10)
    assert res["backend"] == "maxtext"

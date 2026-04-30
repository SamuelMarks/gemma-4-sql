"""Tests for JAX logging."""

from gemma_4_sql.backends.jax.logging import log_metrics


def test_log_metrics() -> None:
    """Test JAX logging."""
    metrics = {"loss": 0.5, "acc": 0.9}
    res = log_metrics(metrics, step=10)
    assert res["backend"] == "jax"

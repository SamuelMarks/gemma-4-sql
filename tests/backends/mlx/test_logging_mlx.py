"""Module docstring."""

from unittest.mock import patch

from gemma_4_sql.backends.mlx.logging import log_metrics


def test_log_metrics_success():
    """Docstring for test_log_metrics_success."""
    metrics = {"loss": 0.5}
    with patch("gemma_4_sql.backends.mlx.logging.SummaryWriter"):
        res = log_metrics(metrics, 10, "logs")
        assert res["status"] == "success"
        assert res["backend"] == "mlx"


def test_log_metrics_missing():
    """Docstring for test_log_metrics_missing."""
    with patch("gemma_4_sql.backends.mlx.logging.SummaryWriter", None):
        res = log_metrics({"loss": 0.5}, 10, "logs")
        assert res["status"] == "mocked_missing_tensorboard"

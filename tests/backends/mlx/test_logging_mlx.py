"""Tests for mlx logging."""

import sys
from unittest.mock import patch


def test_mlx_logging_imports():
    """Test mlx logging imports fallback."""

    with patch.dict(sys.modules, {"mlx": None, "mlx.utils": None, "mlx.utils.tensorboard": None}):
        import gemma_4_sql.backends.mlx.logging as logging_module

        assert logging_module.SummaryWriter is None


def test_log_metrics():
    """Test log_metrics."""
    import gemma_4_sql.backends.mlx.logging as logging_module

    with patch("gemma_4_sql.backends.mlx.logging.log_metrics_wrapper") as mock_wrapper:
        mock_wrapper.return_value = {"status": "ok"}

        res = logging_module.log_metrics({"m": 1.0}, 1, "dir")

        assert res == {"status": "ok"}
        mock_wrapper.assert_called_once_with(backend_name="mlx", metrics={"m": 1.0}, step=1, log_dir="dir", summary_writer_cls=logging_module.SummaryWriter)


def test_mlx_logging_successful_imports():
    """Test mlx logging successful imports."""
    import sys
    from unittest.mock import MagicMock

    mock_mlx = MagicMock()
    mock_mlx.utils = MagicMock()
    mock_mlx.utils.tensorboard = MagicMock()
    mock_mlx.utils.tensorboard.SummaryWriter = MagicMock()

    with patch.dict(
        sys.modules,
        {
            "mlx": mock_mlx,
            "mlx.utils": mock_mlx.utils,
            "mlx.utils.tensorboard": mock_mlx.utils.tensorboard,
        },
    ):
        if "gemma_4_sql.backends.mlx.logging" in sys.modules:
            del sys.modules["gemma_4_sql.backends.mlx.logging"]
        import gemma_4_sql.backends.mlx.logging as logging_module

        assert logging_module.SummaryWriter is not None

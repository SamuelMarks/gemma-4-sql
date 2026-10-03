from unittest.mock import MagicMock, patch

from gemma_4_sql.backends.jax.logging import log_metrics


def test_log_metrics():
    with patch("gemma_4_sql.backends.jax.logging.log_metrics_wrapper") as mock_wrapper:
        mock_wrapper.return_value = {"status": "success"}

        res = log_metrics({"loss": 0.5}, 10, "mylogs")

        assert res == {"status": "success"}
        mock_wrapper.assert_called_once()
        _args, kwargs = mock_wrapper.call_args
        assert kwargs["backend_name"] == "jax"
        assert kwargs["metrics"] == {"loss": 0.5}
        assert kwargs["step"] == 10
        assert kwargs["log_dir"] == "mylogs"
        assert kwargs["extra_fields"] == {"action": "log_metrics"}


def test_log_metrics_success_import(monkeypatch):
    mock_tbx = MagicMock()
    mock_tbx.SummaryWriter = "MockWriter"
    import sys

    sys.modules["tensorboardX"] = mock_tbx

    # Reload the module to trigger the try block
    import importlib

    import gemma_4_sql.backends.jax.logging as log_mod

    importlib.reload(log_mod)

    assert log_mod.SummaryWriter == "MockWriter"

    with patch("gemma_4_sql.backends.jax.logging.log_metrics_wrapper") as mock_wrapper:
        mock_wrapper.return_value = {"status": "success"}
        res = log_mod.log_metrics({"loss": 0.5}, 10, "mylogs")
        assert res == {"status": "success"}
    del sys.modules["tensorboardX"]
    importlib.reload(log_mod)

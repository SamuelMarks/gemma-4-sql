from unittest.mock import MagicMock

import pytest

from gemma_4_sql.backends.common_data import _load_duckdb_dataset


def test_load_duckdb_dataset_missing(monkeypatch):
    monkeypatch.setattr("gemma_4_sql.backends.lazy_loader.LazyLoader.get_module", lambda x: None)
    with pytest.raises(RuntimeError, match="duckdb is required"):
        _load_duckdb_dataset("test.db", "test_table")


def test_load_duckdb_dataset_error(monkeypatch):
    mock_duckdb = MagicMock()
    mock_duckdb.connect.side_effect = Exception("Test DB error")
    monkeypatch.setattr("gemma_4_sql.backends.lazy_loader.LazyLoader.get_module", lambda x: mock_duckdb)
    with pytest.raises(RuntimeError, match="DuckDB error"):
        _load_duckdb_dataset("test.db", "test_table")


from gemma_4_sql.backends.common_dpo import generic_run_training_epochs
from gemma_4_sql.backends.common_logging import log_metrics_wrapper
from gemma_4_sql.backends.common_quantize import quantize_model_wrapper
from gemma_4_sql.backends.common_serve import serve_model_wrapper


def test_common_dpo_run_training_epochs():
    class MockState:
        def __init__(self):
            self.dataloader = [[{"inputs": 1}]]
            self.epochs = 1
            self.policy_model = "policy"
            self.ref_model = "ref"
            self.optimizer = "opt"
            self.beta = 0.1

    class MockLoss:
        def item(self):
            return 1.0

    def mock_step(policy, ref, opt, batch, beta):
        return MockLoss()

    loss = generic_run_training_epochs(MockState(), mock_step)
    assert loss == 1.0


def test_common_logging_close():
    from unittest.mock import Mock

    mock_writer = Mock()
    mock_writer.close = Mock()
    mock_cls = Mock(return_value=mock_writer)

    res = log_metrics_wrapper("test", {"loss": 1.0}, 1, "logs", mock_cls)
    mock_writer.close.assert_called_once()
    assert res["status"] == "success"


def test_common_quantize():
    res1 = quantize_model_wrapper("test", "model", "int8", True, "missing", lambda: (0.5, "quantized"))
    assert res1["status"] == "missing"

    res2 = quantize_model_wrapper("test", "model", "int8", False, "", lambda: (0.5, "quantized"))
    assert res2["status"] == "quantized"


def test_common_serve_missing():
    res = serve_model_wrapper("test", "model", 8000, 32, True, "missing", lambda: None)
    assert res["status"] == "missing"


from gemma_4_sql.backends.common_quantize import apply_bits_and_bytes_quantization


def test_quantize_missing_bitsandbytes():
    res = apply_bits_and_bytes_quantization("int8", None, None)
    assert res[1] == "mocked_missing_bitsandbytes"

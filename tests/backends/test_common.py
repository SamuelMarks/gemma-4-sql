"""Tests for test common module."""

from unittest.mock import MagicMock

import pytest

from gemma_4_sql.backends.common_data import _load_duckdb_dataset


def test_load_duckdb_dataset_missing(monkeypatch):
    """Test load duckdb dataset missing functionality."""
    monkeypatch.setattr("gemma_4_sql.backends.lazy_loader.LazyLoader.get_module", lambda x: None)
    with pytest.raises(RuntimeError, match="duckdb is required"):
        _load_duckdb_dataset("test.db", "test_table")


def test_load_duckdb_dataset_error(monkeypatch):
    """Test load duckdb dataset error functionality."""
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
    """Test common dpo run training epochs functionality."""

    class MockState:
        """Test class for MockState."""

        def __init__(self):
            """Initialize __init__."""
            self.dataloader = [[{"inputs": 1}]]
            self.epochs = 1
            self.policy_model = "policy"
            self.ref_model = "ref"
            self.optimizer = "opt"
            self.beta = 0.1

    class MockLoss:
        """Test class for MockLoss."""

        def item(self):
            """Execute item helper."""
            return 1.0

    def mock_step(policy, ref, opt, batch, beta):
        """Execute mock step helper."""
        return MockLoss()

    import pytest

    loss = generic_run_training_epochs(MockState(), mock_step)
    assert loss == pytest.approx(1.0)


def test_common_logging_close():
    """Test common logging close functionality."""
    from unittest.mock import Mock

    mock_writer = Mock()
    mock_writer.close = Mock()
    mock_cls = Mock(return_value=mock_writer)

    res = log_metrics_wrapper("test", {"loss": 1.0}, 1, "logs", mock_cls, step_duration_s=0.5)
    mock_writer.close.assert_called_once()
    assert res["status"] == "success"
    assert res["step_duration_s"] == 0.5


def test_common_logging_file_fallback(tmp_path):
    """Test common logging file persistence when TensorBoard is absent."""
    import json

    log_dir = str(tmp_path / "fallback_logs")
    res = log_metrics_wrapper(
        "test_backend",
        {"accuracy": 0.98},
        step=5,
        log_dir=log_dir,
        summary_writer_cls=None,
        extra_fields={"epoch": "2"},
        step_duration_s=1.25,
    )
    assert res["status"] == "mocked_missing_tensorboard"
    assert "fallback_file" in res

    metrics_file = tmp_path / "fallback_logs" / "metrics.jsonl"
    assert metrics_file.exists()
    lines = metrics_file.read_text().strip().split("\n")
    data = json.loads(lines[-1])
    assert data["backend"] == "test_backend"
    assert data["step"] == 5
    assert data["metrics"] == {"accuracy": 0.98}
    assert data["epoch"] == "2"
    assert data["step_duration_s"] == 1.25

    class WriterWithClose:
        """Writer with close method."""

        def __init__(self, log_dir: str) -> None:
            """Initialize writer."""

        def add_scalar(self, k: str, v: float, step: int) -> None:
            """Add scalar."""

        def close(self) -> None:
            """Close writer."""

    class WriterNoClose:
        """Writer without close method."""

        def __init__(self, log_dir: str) -> None:
            """Initialize writer."""

        def add_scalar(self, k: str, v: float, step: int) -> None:
            """Add scalar."""

    res_with_close = log_metrics_wrapper("test", {"loss": 0.5}, 1, log_dir=log_dir, summary_writer_cls=WriterWithClose)
    assert res_with_close["status"] == "success"

    res_no_close = log_metrics_wrapper("test", {"loss": 0.5}, 1, log_dir=log_dir, summary_writer_cls=WriterNoClose)
    assert res_no_close["status"] == "success"


def test_common_quantize():
    """Test common quantize functionality."""
    res1 = quantize_model_wrapper("test", "model", "int8", True, "missing", lambda: (0.5, "quantized"))
    assert res1["status"] == "missing"

    res2 = quantize_model_wrapper("test", "model", "int8", False, "", lambda: (0.5, "quantized"))
    assert res2["status"] == "quantized"


def test_common_serve_missing():
    """Test common serve missing functionality."""
    res = serve_model_wrapper("test", "model", 8000, 32, True, "missing", lambda: None)
    assert res["status"] == "missing"


from gemma_4_sql.backends.common_quantize import apply_bits_and_bytes_quantization


def test_quantize_missing_bitsandbytes():
    """Test quantize missing bitsandbytes functionality."""
    res = apply_bits_and_bytes_quantization("int8", None, None)
    assert res[1] == "mocked_missing_bitsandbytes"


def test_apply_bits_and_bytes_quantization():
    """Test int8, int4, skip modules, and unsupported methods in apply_bits_and_bytes_quantization."""
    from gemma_4_sql.exceptions import DependencyMissingError

    called_kwargs: list[dict] = []

    class MockConfig:
        """Test class for MockConfig."""

        def __init__(self, **kwargs):
            """Initialize __init__."""
            called_kwargs.append(kwargs)

    # Test int8 default
    red8, stat8 = apply_bits_and_bytes_quantization("int8", MockConfig, None)
    assert stat8 == "quantized_int8"
    assert red8 == pytest.approx(0.5)
    assert called_kwargs[-1]["load_in_8bit"] is True
    assert called_kwargs[-1]["llm_int8_threshold"] == 6.0

    # Test int8 with skip modules
    apply_bits_and_bytes_quantization("int8", MockConfig, None, llm_int8_skip_modules=["lm_head"])
    assert called_kwargs[-1]["llm_int8_skip_modules"] == ["lm_head"]

    # Test int4
    red4, stat4 = apply_bits_and_bytes_quantization("int4", MockConfig, "float16", bnb_4bit_quant_type="fp4")
    assert stat4 == "quantized_int4"
    assert red4 == pytest.approx(0.75)
    assert called_kwargs[-1]["load_in_4bit"] is True
    assert called_kwargs[-1]["bnb_4bit_quant_type"] == "fp4"

    # Test unsupported method returns unsupported status
    red_unsupp, stat_unsupp = apply_bits_and_bytes_quantization("awq", MockConfig, None)
    assert stat_unsupp == "unsupported_method_awq"
    assert red_unsupp == 0.0

    # Test raise_if_missing
    with pytest.raises(DependencyMissingError, match="bitsandbytes and transformers are required"):
        apply_bits_and_bytes_quantization("int8", None, None, raise_if_missing=True)

    # Test model with config attribute
    class ModelWithConfig:
        """Model with config."""

        class Config:
            """Inner config."""

            quantization_config = None

        config = Config()

    m_with_cfg = ModelWithConfig()
    apply_bits_and_bytes_quantization("int8", MockConfig, None, model=m_with_cfg)
    assert getattr(m_with_cfg, "_is_quantized", False) is True

    # Test model without config attribute
    class ModelWithoutConfig:
        """Model without config."""

    m_no_cfg = ModelWithoutConfig()
    apply_bits_and_bytes_quantization("int8", MockConfig, None, model=m_no_cfg)
    assert getattr(m_no_cfg, "_is_quantized", False) is True


def test_quantize_model_wrapper_error_handling():
    """Test error handling in quantize_model_wrapper."""

    def fail_apply():
        raise RuntimeError("Quantization execution failed")

    res = quantize_model_wrapper("test", "model", "int8", False, "", fail_apply)
    assert "failed: Quantization execution failed" in res["status"]
    assert res["memory_reduction_factor"] == 0.0


def test_serve_model_wrapper_run_server(monkeypatch):
    """Test serve_model_wrapper with run_server=True and test_mode=False."""
    import gemma_4_sql.backends.common_serve as cs

    mock_uvicorn = MagicMock()
    monkeypatch.setattr(cs, "uvicorn", mock_uvicorn)
    monkeypatch.setattr(cs, "FastAPI", MagicMock())

    res = serve_model_wrapper(
        backend_name="test",
        model_name="model",
        port=8000,
        max_batch_size=32,
        missing_deps=False,
        missing_status="",
        app_factory=lambda: "mock_app",
        test_mode=False,
        run_server=True,
    )
    assert res["status"] == "running_test_serve"
    mock_uvicorn.run.assert_called_once_with("mock_app", host="0.0.0.0", port=8000)

    # Test test_mode=False but run_server=False
    mock_uvicorn.reset_mock()
    res2 = serve_model_wrapper(
        backend_name="test",
        model_name="model",
        port=8000,
        max_batch_size=32,
        missing_deps=False,
        missing_status="",
        app_factory=lambda: "mock_app",
        test_mode=False,
        run_server=False,
    )
    assert res2["status"] == "running_test_serve"
    mock_uvicorn.run.assert_not_called()

    # Test test_mode=True
    res3 = serve_model_wrapper(
        backend_name="test",
        model_name="model",
        port=8000,
        max_batch_size=32,
        missing_deps=False,
        missing_status="",
        app_factory=lambda: "mock_app",
        test_mode=True,
    )
    assert res3["status"] == "running_test_serve"


def test_load_duckdb_dataset_invalid_table():
    """Test _load_duckdb_dataset rejects invalid table names."""
    from gemma_4_sql.backends.common_data import _load_duckdb_dataset

    with pytest.raises(RuntimeError, match="Invalid or unsafe table name"):
        _load_duckdb_dataset(":memory:", "drop table users; --")

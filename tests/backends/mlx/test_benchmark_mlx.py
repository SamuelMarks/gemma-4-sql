import importlib
import sys
from unittest.mock import MagicMock

import pytest

import gemma_4_sql.backends.mlx.benchmark as mlx_benchmark
from gemma_4_sql.backends.mlx.benchmark import (
    _get_peak_memory_mb,
    _load_mlx_model_and_device,
    _run_benchmark_pass,
    _sync_and_eval,
    benchmark_model,
)
from gemma_4_sql.exceptions import DependencyMissingError


@pytest.fixture(autouse=True)
def mock_mlx(monkeypatch):
    mock_mx = MagicMock()
    mock_load = MagicMock()
    monkeypatch.setattr(mlx_benchmark, "mx", mock_mx)
    monkeypatch.setattr(mlx_benchmark, "load", mock_load)
    return mock_mx, mock_load


def test_load_mlx_model_missing_deps(monkeypatch):
    monkeypatch.setattr(mlx_benchmark, "mx", None)
    with pytest.raises(DependencyMissingError, match="MLX dependencies"):
        _load_mlx_model_and_device("model", "gpu")


def test_load_mlx_model_cpu(mock_mlx):
    mock_mx, mock_load = mock_mlx
    mock_mx.cpu = MagicMock()
    mock_load.return_value = (MagicMock(), MagicMock())

    _model, device = _load_mlx_model_and_device("model", "cpu")
    assert device == "cpu"
    mock_mx.set_default_device.assert_called_once()
    mock_mx.Device.assert_called_with(mock_mx.cpu)


def test_load_mlx_model_gpu(mock_mlx):
    mock_mx, mock_load = mock_mlx
    mock_mx.gpu = 0
    mock_load.return_value = MagicMock()  # not a tuple

    _model, device = _load_mlx_model_and_device("model", "gpu")
    assert device == "gpu"
    mock_mx.set_default_device.assert_called_once()
    mock_mx.Device.assert_called_with(0)


def test_load_mlx_model_device_error(mock_mlx):
    mock_mx, mock_load = mock_mlx
    mock_mx.set_default_device.side_effect = ValueError("Invalid device")
    mock_load.return_value = MagicMock()

    _model, device = _load_mlx_model_and_device("model", "gpu")
    assert device == "gpu"


def test_sync_and_eval(mock_mlx):
    mock_mx, _ = mock_mlx
    # Test single
    _sync_and_eval(MagicMock())
    mock_mx.eval.assert_called_once()
    mock_mx.eval.reset_mock()

    # Test list
    _sync_and_eval([MagicMock(), MagicMock()])
    mock_mx.eval.assert_called_once()


def test_sync_and_eval_no_mx(monkeypatch):
    monkeypatch.setattr(mlx_benchmark, "mx", None)
    # Should not crash
    _sync_and_eval(MagicMock())


def test_get_peak_memory_mb_no_mx(monkeypatch):
    monkeypatch.setattr(mlx_benchmark, "mx", None)
    assert _get_peak_memory_mb() == 0.0


def test_get_peak_memory_mb_metal(mock_mlx):
    mock_mx, _ = mock_mlx
    mock_mx.metal.is_available.return_value = True
    mock_mx.metal.get_peak_memory.return_value = 1024 * 1024 * 10
    assert _get_peak_memory_mb() == 10.0


def test_get_peak_memory_mb_metal_fallback_active(mock_mlx):
    mock_mx, _ = mock_mlx
    mock_mx.metal.is_available.return_value = True
    mock_mx.metal.get_peak_memory.return_value = 0
    mock_mx.metal.get_active_memory.return_value = 1024 * 1024 * 5
    assert _get_peak_memory_mb() == 5.0


def test_get_peak_memory_mb_no_metal_fallback(mock_mlx):
    mock_mx, _ = mock_mlx
    del mock_mx.metal.get_peak_memory
    del mock_mx.metal.get_active_memory
    mock_mx.get_peak_memory.return_value = 1024 * 1024 * 2
    assert _get_peak_memory_mb() == 2.0


def test_run_benchmark_pass_missing_mx(monkeypatch):
    monkeypatch.setattr(mlx_benchmark, "mx", None)
    with pytest.raises(DependencyMissingError, match="MLX dependencies are missing."):
        _run_benchmark_pass(MagicMock(), 1, 1)


def test_run_benchmark_pass(mock_mlx):
    mock_mx, _ = mock_mlx
    mock_mx.zeros.return_value = MagicMock()
    mock_mx.metal.get_peak_memory.return_value = 100
    mock_model = MagicMock()

    # Test typical success path
    tokens_per_sec, latency_ms, memory_mb = _run_benchmark_pass(model=mock_model, batch_size=1, num_runs=2, prompt_len=2, decode_tokens=2)
    assert tokens_per_sec > 0
    assert latency_ms > 0
    assert memory_mb >= 0


def test_run_benchmark_pass_metal_reset_error(mock_mlx):
    mock_mx, _ = mock_mlx
    mock_mx.metal.reset_peak_memory.side_effect = RuntimeError("Failed")
    mock_mx.zeros.return_value = MagicMock()
    mock_mx.metal.get_peak_memory.return_value = 100
    mock_model = MagicMock()

    # Should not crash
    _run_benchmark_pass(model=mock_model, batch_size=1, num_runs=1, prompt_len=2, decode_tokens=2)


def test_benchmark_model(mock_mlx):
    mock_mx, mock_load = mock_mlx
    mock_load.return_value = MagicMock()
    mock_mx.metal.get_peak_memory.return_value = 100

    result = benchmark_model("model", "cpu", 1, num_runs=1, prompt_len=2, decode_tokens=2)
    assert result["backend"] == "mlx"
    assert "status" in result


def test_benchmark_model_missing_deps(monkeypatch):
    monkeypatch.setattr(mlx_benchmark, "load", None)
    result = benchmark_model("model", "cpu", 1)
    assert result["status"] == "mocked_missing_mlx"


def test_get_peak_memory_mb_no_methods(mock_mlx):
    mock_mx, _ = mock_mlx
    del mock_mx.metal.get_peak_memory
    del mock_mx.metal.get_active_memory
    del mock_mx.get_peak_memory
    assert _get_peak_memory_mb() == 0.0


def test_run_benchmark_pass_no_model(mock_mlx):
    mock_mx, _ = mock_mlx
    mock_mx.zeros.return_value = MagicMock()
    del mock_mx.metal.get_peak_memory
    del mock_mx.metal.get_active_memory
    mock_mx.get_peak_memory.return_value = 0

    tokens_per_sec, latency_ms, memory_mb = _run_benchmark_pass(model=None, batch_size=1, num_runs=1, prompt_len=2, decode_tokens=2)
    assert tokens_per_sec > 0
    assert latency_ms > 0
    assert memory_mb == 0.0


def test_load_mlx_model_no_device_support(mock_mlx):
    mock_mx, mock_load = mock_mlx
    del mock_mx.set_default_device
    mock_load.return_value = MagicMock()

    _model, device = _load_mlx_model_and_device("model", "gpu")
    assert device == "gpu"


def test_get_peak_memory_mb_metal_no_peak_memory(mock_mlx):
    mock_mx, _ = mock_mlx
    mock_mx.metal.is_available.return_value = True
    del mock_mx.metal.get_peak_memory
    mock_mx.get_peak_memory.return_value = 0
    mock_mx.metal.get_active_memory.return_value = 1024 * 1024 * 3
    assert _get_peak_memory_mb() == 3.0


def test_get_peak_memory_mb_metal_no_active_memory(mock_mlx):
    mock_mx, _ = mock_mlx
    mock_mx.metal.is_available.return_value = True
    del mock_mx.metal.get_peak_memory
    del mock_mx.metal.get_active_memory
    mock_mx.get_peak_memory.return_value = 1024 * 1024 * 4
    assert _get_peak_memory_mb() == 4.0


def test_benchmark_module_reload():
    import gemma_4_sql.backends.mlx.benchmark as bm

    importlib.reload(bm)


def test_benchmark_module_reload_with_mock_modules():
    import gemma_4_sql.backends.mlx.benchmark as bm

    mock_mlx = MagicMock()
    mock_mlx_lm = MagicMock()
    mock_mlx_lm.load = "mocked_load"

    # Save original modules
    orig_mlx = sys.modules.get("mlx.core")
    orig_mlx_lm = sys.modules.get("mlx_lm")

    sys.modules["mlx.core"] = mock_mlx
    sys.modules["mlx_lm"] = mock_mlx_lm
    try:
        importlib.reload(bm)
    finally:
        if orig_mlx:
            sys.modules["mlx.core"] = orig_mlx
        else:
            del sys.modules["mlx.core"]

        if orig_mlx_lm:
            sys.modules["mlx_lm"] = orig_mlx_lm
        else:
            del sys.modules["mlx_lm"]


def test_get_peak_memory_mb_no_metal_but_has_get_peak(mock_mlx):
    mock_mx, _ = mock_mlx
    del mock_mx.metal
    mock_mx.get_peak_memory.return_value = 1024 * 1024 * 10
    assert _get_peak_memory_mb() == 10.0


def test_run_benchmark_pass_no_metal_module(mock_mlx):
    mock_mx, _ = mock_mlx
    mock_mx.zeros.return_value = MagicMock()
    del mock_mx.metal
    mock_mx.get_peak_memory.return_value = 100
    mock_model = MagicMock()

    _tokens_per_sec, _latency_ms, memory_mb = _run_benchmark_pass(model=mock_model, batch_size=1, num_runs=2, prompt_len=2, decode_tokens=2)
    assert memory_mb > 0

"""Tests for MLX benchmarking and Metal memory profiling."""

from __future__ import annotations

from unittest.mock import MagicMock

import mlx.core as mx
import pytest

import gemma_4_sql.backends.mlx.benchmark as bm
from gemma_4_sql.backends.mlx.benchmark import (
    _get_peak_memory_mb,
    _load_mlx_model_and_device,
    _run_benchmark_pass,
    _sync_and_eval,
    benchmark_model,
)
from gemma_4_sql.exceptions import DependencyMissingError


def test_sync_and_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _sync_and_eval with single and sequence of tensors and None mx."""
    t1 = mx.zeros((2, 2))
    _sync_and_eval(t1)

    t2 = mx.ones((2, 2))
    _sync_and_eval([t1, t2])
    _sync_and_eval((t1, t2))

    monkeypatch.setattr(bm, "mx", None)
    _sync_and_eval(t1)


def test_get_peak_memory_mb(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Metal peak memory retrieval branches."""
    # When mx is None
    monkeypatch.setattr(bm, "mx", None)
    assert _get_peak_memory_mb() == 0.0

    # When mx.metal is available with peak memory
    class MockMetalWithPeak:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_peak_memory() -> int:
            return 1024 * 1024 * 50  # 50 MB

    mock_mx = type("MockMX", (), {"metal": MockMetalWithPeak()})
    monkeypatch.setattr(bm, "mx", mock_mx)
    assert _get_peak_memory_mb() == pytest.approx(50.0)

    # When peak memory is 0, fall back to get_active_memory
    class MockMetalWithActive:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_peak_memory() -> int:
            return 0

        @staticmethod
        def get_active_memory() -> int:
            return 1024 * 1024 * 25  # 25 MB

    mock_mx_active = type("MockMX", (), {"metal": MockMetalWithActive()})
    monkeypatch.setattr(bm, "mx", mock_mx_active)
    assert _get_peak_memory_mb() == pytest.approx(25.0)

    # When metal has active memory but NO get_peak_memory method
    class MockMetalOnlyActive:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_active_memory() -> int:
            return 1024 * 1024 * 15

    mock_mx_only_active = type("MockMX", (), {"metal": MockMetalOnlyActive()})
    monkeypatch.setattr(bm, "mx", mock_mx_only_active)
    assert _get_peak_memory_mb() == pytest.approx(15.0)

    # When peak memory is 0 and no get_active_memory
    class MockMetalZeroOnly:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_peak_memory() -> int:
            return 0

    mock_mx_zero = type("MockMX", (), {"metal": MockMetalZeroOnly()})
    monkeypatch.setattr(bm, "mx", mock_mx_zero)
    assert _get_peak_memory_mb() == 0.0

    # When metal is not available and no get_peak_memory on mx
    class MockMetalNotAvail:
        @staticmethod
        def is_available() -> bool:
            return False

    mock_mx_no_metal = type("MockMX", (), {"metal": MockMetalNotAvail()})
    monkeypatch.setattr(bm, "mx", mock_mx_no_metal)
    assert _get_peak_memory_mb() == 0.0

    # Fallback to mx.get_peak_memory
    mock_mx_direct = type("MockMX", (), {"get_peak_memory": lambda: 1024 * 1024 * 10})
    monkeypatch.setattr(bm, "mx", mock_mx_direct)
    assert _get_peak_memory_mb() == pytest.approx(10.0)


def test_load_mlx_model_and_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test model loading and device selection."""
    # Test mode flag
    model, dev = _load_mlx_model_and_device("model", "gpu", test_mode=True)
    assert model is None
    assert dev == "cpu"

    # Missing dependencies
    monkeypatch.setattr(bm, "load", None)
    with pytest.raises(DependencyMissingError, match="MLX dependencies"):
        _load_mlx_model_and_device("model", "gpu")

    # CPU vs GPU device with single loaded object
    dummy_model = MagicMock()
    monkeypatch.setattr(bm, "load", lambda _name: dummy_model)

    _m_gpu, dev_gpu = _load_mlx_model_and_device("model", "gpu")
    assert dev_gpu == "gpu"

    _m_cpu, dev_cpu = _load_mlx_model_and_device("model", "cpu")
    assert dev_cpu == "cpu"

    # Device error handling
    monkeypatch.setattr(bm.mx, "set_default_device", MagicMock(side_effect=RuntimeError("Device error")))
    _m_err, dev_err = _load_mlx_model_and_device("model", "gpu")
    assert dev_err == "gpu"

    # When mx has no set_default_device
    mock_mx_no_set = type("MockMXNoSet", (), {})()
    monkeypatch.setattr(bm, "mx", mock_mx_no_set)
    _m_no_set, dev_no_set = _load_mlx_model_and_device("model", "gpu")
    assert dev_no_set == "gpu"


def test_run_benchmark_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test warm-up and timed benchmark loop."""
    step = 0

    def mock_model(inputs: mx.array) -> mx.array:
        nonlocal step
        step += 1
        return inputs * 2.0

    # Missing mx dependency
    monkeypatch.setattr(bm, "mx", None)
    with pytest.raises(DependencyMissingError, match="MLX dependencies"):
        _run_benchmark_pass(mock_model, batch_size=2, num_runs=2)

    # Normal run with model
    monkeypatch.setattr(bm, "mx", mx)
    tps, lat, mem = _run_benchmark_pass(
        model=mock_model,
        batch_size=2,
        num_runs=2,
        prompt_len=8,
        decode_tokens=4,
    )
    assert tps > 0.0
    assert lat > 0.0
    assert mem >= 0.0

    # Run with model=None
    tps_none, _lat_none, _mem_none = _run_benchmark_pass(
        model=None,
        batch_size=2,
        num_runs=1,
        prompt_len=4,
        decode_tokens=2,
    )
    assert tps_none > 0.0

    # Reset peak memory successful
    class MockMetalResetSuccess:
        @staticmethod
        def reset_peak_memory() -> None:
            pass

    monkeypatch.setattr(mx, "metal", MockMetalResetSuccess())
    tps_reset, _, _ = _run_benchmark_pass(
        model=mock_model,
        batch_size=1,
        num_runs=1,
        prompt_len=4,
        decode_tokens=2,
    )
    assert tps_reset > 0.0

    # Reset peak memory error branch
    class MockMetalResetError:
        @staticmethod
        def reset_peak_memory() -> None:
            raise RuntimeError("Reset failed")

    monkeypatch.setattr(bm.mx, "metal", MockMetalResetError())
    tps2, _lat2, _mem2 = _run_benchmark_pass(
        model=mock_model,
        batch_size=1,
        num_runs=1,
        prompt_len=4,
        decode_tokens=2,
    )
    assert tps2 > 0.0

    # Metal with no reset_peak_memory method
    class MockMetalNoReset:
        pass

    monkeypatch.setattr(bm.mx, "metal", MockMetalNoReset())
    tps3, _lat3, _mem3 = _run_benchmark_pass(
        model=mock_model,
        batch_size=1,
        num_runs=1,
        prompt_len=4,
        decode_tokens=2,
    )
    assert tps3 > 0.0


def test_benchmark_model_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test benchmark_model wrapper for success, missing deps, and errors."""
    mock_model = lambda x: x + 1.0
    monkeypatch.setattr(bm, "load", lambda _name: mock_model)

    res = benchmark_model("test_mlx", "gpu", batch_size=2, num_runs=2, prompt_len=4, decode_tokens=2)
    assert res["status"] == "success"
    assert res["tokens_per_sec"] > 0.0
    assert res["latency_ms"] > 0.0

    # Missing dependency
    monkeypatch.setattr(bm, "mx", None)
    res_missing = benchmark_model("test_mlx", "gpu", batch_size=2)
    assert res_missing["status"] == "mocked_missing_mlx"

    # Runtime error during execution
    monkeypatch.setattr(bm, "mx", mx)
    monkeypatch.setattr(bm, "load", MagicMock(side_effect=RuntimeError("Load failure")))
    res_err = benchmark_model("test_mlx", "gpu", batch_size=2)
    assert "failed" in res_err["status"]

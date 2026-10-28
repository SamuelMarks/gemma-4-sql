"""Tests for MLX benchmark."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import gemma_4_sql.backends.mlx.benchmark as benchmark_module
from gemma_4_sql.exceptions import DependencyMissingError
from tests.conftest import requires_mlx

pytestmark = requires_mlx


def test_mlx_benchmark_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test mlx benchmark missing deps."""
    monkeypatch.setattr(benchmark_module, "mx", None)
    monkeypatch.setattr(benchmark_module, "load", None)

    with pytest.raises(DependencyMissingError, match="MLX dependencies"):
        benchmark_module._load_mlx_model_and_device("model", "cpu")

    with pytest.raises(DependencyMissingError, match="MLX dependencies"):
        benchmark_module._run_benchmark_pass(None, 1, 1)


def test_load_mlx_model_and_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test load mlx model and device."""
    mock_mx = MagicMock()
    mock_load = MagicMock(return_value=["model", "tokenizer"])
    monkeypatch.setattr(benchmark_module, "mx", mock_mx)
    monkeypatch.setattr(benchmark_module, "load", mock_load)

    model, device = benchmark_module._load_mlx_model_and_device("model", "cpu")
    assert model == "model"
    assert device == "cpu"
    mock_mx.set_default_device.assert_called()

    model, device = benchmark_module._load_mlx_model_and_device("model", "gpu")
    assert device == "gpu"

    mock_load.return_value = "model_only"
    model, device = benchmark_module._load_mlx_model_and_device("model", "gpu")
    assert model == "model_only"

    mock_mx.set_default_device.side_effect = ValueError("error")
    model, device = benchmark_module._load_mlx_model_and_device("model", "gpu")

    del mock_mx.set_default_device
    model, device = benchmark_module._load_mlx_model_and_device("model", "gpu")
    assert device == "gpu"
    assert model == "model_only"


def test_sync_and_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test sync and eval."""
    mock_mx = MagicMock()
    monkeypatch.setattr(benchmark_module, "mx", mock_mx)

    benchmark_module._sync_and_eval(["t1", "t2"])
    mock_mx.eval.assert_called_with("t1", "t2")

    benchmark_module._sync_and_eval("t1")
    mock_mx.eval.assert_called_with("t1")

    monkeypatch.setattr(benchmark_module, "mx", None)
    benchmark_module._sync_and_eval("t1")


def test_get_peak_memory_mb(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test get peak memory mb across Metal and fallback branches."""
    mock_mx = MagicMock()
    monkeypatch.setattr(benchmark_module, "mx", mock_mx)

    mock_mx.metal.is_available.return_value = True
    mock_mx.metal.get_peak_memory.return_value = 1048576 * 10
    assert benchmark_module._get_peak_memory_mb() == 10.0

    mock_mx.metal.get_peak_memory.return_value = 0
    mock_mx.metal.get_active_memory.return_value = 1048576 * 5
    assert benchmark_module._get_peak_memory_mb() == 5.0

    del mock_mx.metal.get_active_memory
    mock_mx.get_peak_memory.return_value = 1048576 * 2
    assert benchmark_module._get_peak_memory_mb() == 2.0

    mock_mx.metal.is_available.return_value = False
    mock_mx.get_peak_memory.return_value = 1048576 * 2
    assert benchmark_module._get_peak_memory_mb() == 2.0

    del mock_mx.get_peak_memory
    assert benchmark_module._get_peak_memory_mb() == 0.0

    monkeypatch.setattr(benchmark_module, "mx", None)
    assert benchmark_module._get_peak_memory_mb() == 0.0


def test_get_peak_memory_mb_missing_metal_get_peak(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test get_peak_memory_mb missing metal.get_peak_memory to hit branch."""
    mock_mx = MagicMock()
    monkeypatch.setattr(benchmark_module, "mx", mock_mx)
    mock_mx.metal.is_available.return_value = True
    del mock_mx.metal.get_peak_memory
    mock_mx.metal.get_active_memory.return_value = 1048576 * 3
    assert benchmark_module._get_peak_memory_mb() == 3.0


@patch("time.perf_counter")
def test_run_benchmark_pass(mock_perf: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test run benchmark pass."""
    mock_mx = MagicMock()
    monkeypatch.setattr(benchmark_module, "mx", mock_mx)

    mock_mx.metal.is_available.return_value = False
    mock_mx.get_peak_memory.return_value = 0
    mock_mx.zeros.return_value = MagicMock()

    mock_model = MagicMock(return_value="out")
    times = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    mock_perf.side_effect = times * 10

    tokens_sec, lat_ms, _mem_mb = benchmark_module._run_benchmark_pass(
        model=mock_model,
        batch_size=1,
        num_runs=1,
        prompt_len=2,
        decode_tokens=2,
    )
    assert tokens_sec > 0
    assert lat_ms > 0

    mock_mx.metal.reset_peak_memory.side_effect = RuntimeError("error")
    benchmark_module._run_benchmark_pass(mock_model, 1, 1, mode="prefill")
    benchmark_module._run_benchmark_pass(mock_model, 1, 1, mode="decode")

    del mock_mx.metal
    benchmark_module._run_benchmark_pass(None, 1, 1)


def test_benchmark_model_behavioral(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify benchmark_model executes real MLX forward passes on a tiny mlx.nn.Module through run_benchmark_wrapper."""
    import mlx.core as mx
    import mlx.nn as mlx_nn

    class TinyMLXBenchModel(mlx_nn.Module):
        """Tiny MLX causal LM for real benchmark execution."""

        def __init__(self) -> None:
            """Initialize TinyMLXBenchModel."""
            super().__init__()
            self.embed = mlx_nn.Embedding(32, 8)
            self.head = mlx_nn.Linear(8, 32)

        def __call__(self, x: mx.array) -> mx.array:
            """Run forward pass on input token IDs."""
            return self.head(self.embed(x))

    tiny_model = TinyMLXBenchModel()
    mx.eval(tiny_model.parameters())
    monkeypatch.setattr(benchmark_module, "load", lambda _m: (tiny_model, "tok"))  # pyright: ignore[reportUnknownLambdaType]

    res = benchmark_module.benchmark_model("model", "cpu", 1, num_runs=1, prompt_len=4, decode_tokens=2)
    assert res["backend"] == "mlx"
    assert res["status"] == "success"
    assert res["tokens_per_sec"] > 0.0
    assert res["latency_ms"] > 0.0

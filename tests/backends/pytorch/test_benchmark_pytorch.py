"""Tests for PyTorch benchmark."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import gemma_4_sql.backends.pytorch.benchmark as benchmark_module
from gemma_4_sql.exceptions import DependencyMissingError
from tests.conftest import requires_torch

pytestmark = requires_torch


def test_pytorch_benchmark_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test pytorch benchmark missing deps."""
    monkeypatch.setattr(benchmark_module, "torch", None)
    monkeypatch.setattr(benchmark_module, "AutoModelForCausalLM", None)

    with pytest.raises(DependencyMissingError, match="PyTorch dependencies"):
        benchmark_module.benchmark_model("model")


def test_get_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _get_device across cpu, cuda, and mps."""
    mock_torch = MagicMock()
    monkeypatch.setattr(benchmark_module, "torch", mock_torch)

    assert benchmark_module._get_device("cpu") == "cpu"

    mock_torch.cuda.is_available.return_value = True
    assert benchmark_module._get_device("gpu") == "cuda"

    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = True
    assert benchmark_module._get_device("gpu") == "mps"

    mock_torch.backends.mps.is_available.return_value = False
    assert benchmark_module._get_device("gpu") == "cpu"


def test_load_pytorch_model_and_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _load_pytorch_model_and_device across native and HF backends."""
    mock_torch = MagicMock()
    mock_auto = MagicMock()
    monkeypatch.setattr(benchmark_module, "torch", mock_torch)
    monkeypatch.setattr(benchmark_module, "AutoModelForCausalLM", mock_auto)

    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = False
    mock_torch.bfloat16 = "bf16"
    mock_torch.float32 = "fp32"

    with patch("gemma_4_sql.backends.pytorch.gemma4.modeling.Gemma4ForCausalLM") as mock_native:
        mock_model = MagicMock()
        mock_native.return_value = mock_model

        with patch("gemma_4_sql.backends.pytorch.gemma4.modeling.Gemma4Config"):
            _res, dev = benchmark_module._load_pytorch_model_and_device("test_model", "cpu", dtype="bfloat16", backend_alias="pytorch_native")
            assert dev == "cpu"

            del mock_torch.bfloat16
            del mock_torch.float32
            benchmark_module._load_pytorch_model_and_device("test_model", "cpu", dtype="unknown", backend_alias="pytorch_native")

            mock_torch.bfloat16 = None
            benchmark_module._load_pytorch_model_and_device("test_model", "cpu", dtype="bfloat16", backend_alias="pytorch_native")
            del mock_torch.bfloat16

            del mock_model.to
            benchmark_module._load_pytorch_model_and_device("test_model", "cpu", dtype="unknown", backend_alias="pytorch_native")

    mock_hf_model = MagicMock()
    mock_auto.from_pretrained.return_value = mock_hf_model
    benchmark_module._load_pytorch_model_and_device("model", "cpu", dtype="bfloat16", backend_alias="pytorch")

    mock_torch.compile.side_effect = RuntimeError("compile error")
    benchmark_module._load_pytorch_model_and_device("model", "cpu", dtype="bfloat16", backend_alias="pytorch")

    del mock_hf_model.to
    del mock_hf_model.eval
    del mock_torch.compile
    benchmark_module._load_pytorch_model_and_device("model", "cpu", dtype="bfloat16", backend_alias="pytorch")


def test_sync_and_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _sync_cuda and _get_memory_mb."""
    mock_torch = MagicMock()
    monkeypatch.setattr(benchmark_module, "torch", mock_torch)

    benchmark_module._sync_cuda("cuda")
    mock_torch.cuda.synchronize.assert_called_once()

    benchmark_module._sync_cuda("mps")
    mock_torch.mps.synchronize.assert_called_once()

    benchmark_module._sync_cuda("cpu")

    mock_torch.cuda.max_memory_allocated.return_value = 1048576 * 10
    assert benchmark_module._get_memory_mb(None, "cuda") == 10.0

    mock_torch.mps.driver_allocated_memory.return_value = 1048576 * 5
    assert benchmark_module._get_memory_mb(None, "mps") == 5.0

    assert benchmark_module._get_memory_mb(None, "cpu") > 0.0


@patch("time.time")
def test_run_benchmark_pass(mock_time: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _run_benchmark_pass."""
    mock_torch = MagicMock()
    monkeypatch.setattr(benchmark_module, "torch", mock_torch)

    mock_model = MagicMock()
    times = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    mock_time.side_effect = times * 5
    tps, _lat, _mem = benchmark_module._run_benchmark_pass(mock_model, "cuda", 1, 1, 1, "prefill", 10)
    assert tps > 0

    mock_time.side_effect = times * 5
    tps, _lat, _mem = benchmark_module._run_benchmark_pass(mock_model, "cpu", 1, 1, 1, "decode", 10)
    assert tps > 0

    del mock_torch.manual_seed
    del mock_torch.no_grad
    mock_inputs = MagicMock(spec=[])
    mock_torch.randint.return_value = mock_inputs
    mock_time.side_effect = times * 5
    benchmark_module._run_benchmark_pass(mock_model, "cpu", 1, 1, 1, "prefill", 10)


def test_benchmark_model_behavioral(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify benchmark_model executes through run_benchmark_wrapper with real stats."""
    mock_torch = MagicMock()
    monkeypatch.setattr(benchmark_module, "torch", mock_torch)
    monkeypatch.setattr(benchmark_module, "AutoModelForCausalLM", MagicMock())

    with (
        patch.object(benchmark_module, "_load_pytorch_model_and_device", return_value=(MagicMock(), "cpu")),
        patch.object(benchmark_module, "_run_benchmark_pass", return_value=(250.0, 4.0, 128.0)),
    ):
        res = benchmark_module.benchmark_model("model", "cpu", 1)
        assert res["backend"] == "pytorch"
        assert res["status"] == "success"
        assert res["tokens_per_sec"] == 250.0
        assert res["latency_ms"] == 4.0

    mock_model_no_gen = MagicMock(spec=[])
    with patch("time.time", side_effect=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]):
        benchmark_module._run_benchmark_pass(mock_model_no_gen, "cpu", 1, 1, 1, "decode", 10)


def test_real_pytorch_tiny_gemma4_benchmark_cpu() -> None:
    """Verify _run_benchmark_pass executes against a real tiny Gemma4ForCausalLM on CPU."""
    from gemma_4_sql.backends.pytorch.gemma4.config import Gemma4Config
    from gemma_4_sql.backends.pytorch.gemma4.modeling import Gemma4ForCausalLM

    cfg = Gemma4Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=1,
    )
    model = Gemma4ForCausalLM(cfg)
    tps, lat, mem = benchmark_module._run_benchmark_pass(model, "cpu", 1, 1, 1, "prefill", 2)
    assert tps > 0.0
    assert lat > 0.0
    assert mem > 0.0


def test_torch_runtime_adapter_edge_cases() -> None:
    """Verify TorchRuntimeAdapter handles None torch module and float16/float32 dtype resolution."""
    from gemma_4_sql.backends.pytorch.runtime_adapter import TorchRuntimeAdapter

    empty_adapter = TorchRuntimeAdapter(None)
    assert empty_adapter.get_device("gpu") == "cpu"
    assert empty_adapter.resolve_dtype("bfloat16") is None
    assert empty_adapter.long_dtype() is None
    assert empty_adapter.compile_model("m") == "m"
    empty_adapter.synchronize("cuda")
    assert empty_adapter.max_memory_mb("cuda", 32.0) == 32.0
    empty_adapter.reset_peak_memory_stats("cuda")
    empty_adapter.manual_seed(0)
    with empty_adapter.no_grad():
        pass

    mock_t = MagicMock()
    mock_t.float16 = "fp16"
    mock_t.float32 = "fp32"
    mock_t.compile = 123  # non-callable compile attribute
    adapter = TorchRuntimeAdapter(mock_t)
    assert adapter.resolve_dtype("float16") == "fp16"
    assert adapter.resolve_dtype("float32") == "fp32"
    assert adapter.compile_model("m") == "m"

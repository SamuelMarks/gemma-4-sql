"""Tests for PyTorch benchmark."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.exceptions import DependencyMissingError


def test_pytorch_benchmark_imports():
    """Test pytorch benchmark imports fallback."""
    import importlib

    with patch.dict(sys.modules, {"torch": None, "transformers": None}):
        import gemma_4_sql.backends.pytorch.benchmark as benchmark_module

        importlib.reload(benchmark_module)
        assert benchmark_module.torch is None
        assert benchmark_module.AutoModelForCausalLM is None
    importlib.reload(benchmark_module)


def test_get_device():
    """Test _get_device."""
    import gemma_4_sql.backends.pytorch.benchmark as benchmark_module

    assert benchmark_module._get_device("cpu") == "cpu"

    with patch("gemma_4_sql.backends.pytorch.benchmark.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = True
        assert benchmark_module._get_device("gpu") == "cuda"

        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True
        assert benchmark_module._get_device("gpu") == "mps"

        mock_torch.cuda = None
        mock_torch.backends = None
        assert benchmark_module._get_device("gpu") == "cpu"


def test_load_pytorch_model_and_device():
    """Test _load_pytorch_model_and_device."""
    import gemma_4_sql.backends.pytorch.benchmark as benchmark_module

    mock_torch = MagicMock()
    mock_torch.float32 = "f32"
    benchmark_module.torch = mock_torch

    mock_auto = MagicMock()
    mock_model = MagicMock()
    mock_auto.from_pretrained.return_value = mock_model
    benchmark_module.AutoModelForCausalLM = mock_auto

    # default
    model, dev = benchmark_module._load_pytorch_model_and_device("m", "cpu")
    assert model == mock_torch.compile.return_value
    mock_model.to.assert_called_with("cpu")
    mock_model.eval.assert_called_once()
    mock_torch.compile.assert_called_once_with(mock_model)

    # native
    with patch("gemma_4_sql.backends.pytorch.gemma4.modeling.Gemma4ForCausalLM") as mock_native:
        mock_native_model = MagicMock()
        mock_native.return_value = mock_native_model
        with patch("gemma_4_sql.backends.pytorch.gemma4.modeling.Gemma4Config"):
            model, dev = benchmark_module._load_pytorch_model_and_device("m", "cpu", backend_alias="pytorch_native")
            assert model == mock_torch.compile.return_value
            mock_native_model.to.assert_any_call("cpu")
            mock_native_model.eval.assert_called_once()

    # test compile error
    mock_torch.compile.side_effect = RuntimeError("error")
    model, dev = benchmark_module._load_pytorch_model_and_device("m", "cpu")
    # Doesn't raise, just logs warning
    assert model == mock_model

    # native missing to/eval/compile
    with patch("gemma_4_sql.backends.pytorch.gemma4.modeling.Gemma4ForCausalLM") as mock_native:
        mock_native_model = MagicMock(spec=[])
        mock_native.return_value = mock_native_model
        with patch("gemma_4_sql.backends.pytorch.gemma4.modeling.Gemma4Config"):
            benchmark_module.torch.bfloat16 = None
            del benchmark_module.torch.compile
            model, dev = benchmark_module._load_pytorch_model_and_device("m", "cpu", backend_alias="pytorch_native", dtype="missing")

            # test native with to and float32 None
            mock_native_model_2 = MagicMock()
            mock_native.return_value = mock_native_model_2
            benchmark_module.torch.float32 = None
            benchmark_module._load_pytorch_model_and_device("m", "cpu", backend_alias="pytorch_native", dtype="missing")

    # HF missing to/eval
    mock_auto.from_pretrained.return_value = MagicMock(spec=[])
    benchmark_module._load_pytorch_model_and_device("m", "cpu")


def test_sync_cuda():
    """Test _sync_cuda."""
    import gemma_4_sql.backends.pytorch.benchmark as benchmark_module

    mock_torch = MagicMock()
    benchmark_module.torch = mock_torch

    benchmark_module._sync_cuda("cuda")
    mock_torch.cuda.synchronize.assert_called_once()

    benchmark_module._sync_cuda("mps")
    mock_torch.mps.synchronize.assert_called_once()

    benchmark_module._sync_cuda("cpu")


def test_get_memory_mb():
    """Test _get_memory_mb."""
    import gemma_4_sql.backends.pytorch.benchmark as benchmark_module

    mock_torch = MagicMock()
    benchmark_module.torch = mock_torch

    mock_torch.cuda.max_memory_allocated.return_value = 1048576 * 10
    assert benchmark_module._get_memory_mb(None, "cuda") == 10.0

    mock_torch.mps.driver_allocated_memory.return_value = 1048576 * 5
    assert benchmark_module._get_memory_mb(None, "mps") == 5.0

    assert benchmark_module._get_memory_mb(None, "cpu") == 8192.0


@patch("time.time")
def test_run_benchmark_pass(mock_time):
    """Test _run_benchmark_pass."""
    import gemma_4_sql.backends.pytorch.benchmark as benchmark_module

    mock_torch = MagicMock()
    benchmark_module.torch = mock_torch

    mock_model = MagicMock()

    times = [0.0, 0.1]
    mock_time.side_effect = times * 10

    # prefill
    tok, lat, mem = benchmark_module._run_benchmark_pass(mock_model, "cpu", 1, 1, 1, "prefill", 10)
    assert tok > 0
    assert lat > 0
    mock_model.assert_called()

    # generate
    tok, lat, mem = benchmark_module._run_benchmark_pass(mock_model, "cpu", 1, 1, 1, "decode", 10)
    assert tok > 0
    assert lat > 0
    mock_model.generate.assert_called()

    # device cuda
    benchmark_module._run_benchmark_pass(mock_model, "cuda", 1, 1, 1, "prefill", 10)
    mock_torch.cuda.reset_peak_memory_stats.assert_called()

    # Missing hasattr
    del mock_torch.manual_seed
    del mock_torch.no_grad
    mock_torch.randint.return_value = MagicMock(spec=[])
    benchmark_module._run_benchmark_pass(MagicMock(spec=[]), "cpu", 1, 1, 1, "prefill", 10)

    # generate missing hasattr
    benchmark_module._run_benchmark_pass(MagicMock(spec=[]), "cpu", 1, 1, 1, "decode", 10)


def test_benchmark_model():
    """Test benchmark_model."""
    import gemma_4_sql.backends.pytorch.benchmark as benchmark_module

    benchmark_module.torch = MagicMock()
    benchmark_module.AutoModelForCausalLM = MagicMock()

    with patch("gemma_4_sql.backends.pytorch.benchmark.run_benchmark_wrapper") as mock_wrapper:
        mock_wrapper.return_value = {"status": "ok"}

        res = benchmark_module.benchmark_model("m", "cpu", 1)
        assert res == {"status": "ok"}

        fn = mock_wrapper.call_args[1]["benchmark_fn"]
        with patch("gemma_4_sql.backends.pytorch.benchmark._load_pytorch_model_and_device") as mock_load:
            mock_load.return_value = ("model", "cpu")
            with patch("gemma_4_sql.backends.pytorch.benchmark._run_benchmark_pass") as mock_pass:
                mock_pass.return_value = (1.0, 2.0, 3.0)
                assert fn() == (1.0, 2.0, 3.0)

    benchmark_module.torch = None
    with pytest.raises(DependencyMissingError):
        benchmark_module.benchmark_model("m", "cpu", 1)


def test_load_pytorch_model_and_device_torch_dtype_none():
    """Docstring for test_load_pytorch_model_and_device_torch_dtype_none."""
    from unittest.mock import MagicMock, patch

    import gemma_4_sql.backends.pytorch.benchmark as benchmark_module

    mock_torch = MagicMock()
    del mock_torch.missing
    mock_torch.float32 = None
    benchmark_module.torch = mock_torch

    with patch("gemma_4_sql.backends.pytorch.gemma4.modeling.Gemma4ForCausalLM") as mock_native:
        mock_model = MagicMock()
        mock_native.return_value = mock_model
        with patch("gemma_4_sql.backends.pytorch.gemma4.modeling.Gemma4Config"):
            benchmark_module._load_pytorch_model_and_device("m", "cpu", backend_alias="pytorch_native", dtype="missing")
            mock_model.to.assert_called_with("cpu")


def test_run_benchmark_pass_no_generate():
    """Docstring for test_run_benchmark_pass_no_generate."""
    from unittest.mock import MagicMock, patch

    import gemma_4_sql.backends.pytorch.benchmark as benchmark_module

    mock_torch = MagicMock()
    benchmark_module.torch = mock_torch
    mock_model = MagicMock(spec=[])

    with patch("time.time") as mock_time:
        mock_time.side_effect = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        benchmark_module._run_benchmark_pass(mock_model, "cpu", 1, 1, 1, "decode", 10)

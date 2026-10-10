"""Tests for mlx benchmark."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.exceptions import DependencyMissingError


def test_mlx_benchmark_imports():
    """Test mlx benchmark imports fallback."""
    # test mlx core fails
    with patch.dict(sys.modules, {"mlx": None, "mlx.core": None, "mlx_lm": MagicMock()}):
        if "gemma_4_sql.backends.mlx.benchmark" in sys.modules:
            del sys.modules["gemma_4_sql.backends.mlx.benchmark"]
        import gemma_4_sql.backends.mlx.benchmark as benchmark_module

        assert benchmark_module.mx is None

    # test mlx_lm fails
    with patch.dict(sys.modules, {"mlx": MagicMock(), "mlx.core": MagicMock(), "mlx_lm": None}):
        if "gemma_4_sql.backends.mlx.benchmark" in sys.modules:
            del sys.modules["gemma_4_sql.backends.mlx.benchmark"]
        import gemma_4_sql.backends.mlx.benchmark as benchmark_module

        assert benchmark_module.load is None


def test_mlx_benchmark_missing_deps():
    """Test mlx benchmark missing deps."""
    with patch.dict(sys.modules, {"mlx": None, "mlx.core": None, "mlx_lm": None}):
        import gemma_4_sql.backends.mlx.benchmark as benchmark_module

        benchmark_module.mx = None
        benchmark_module.load = None

        with pytest.raises(DependencyMissingError, match="MLX dependencies"):
            benchmark_module._load_mlx_model_and_device("model", "cpu")

        with pytest.raises(DependencyMissingError, match="MLX dependencies"):
            benchmark_module._run_benchmark_pass(None, 1, 1)


def test_load_mlx_model_and_device():
    """Test load mlx model and device."""
    import gemma_4_sql.backends.mlx.benchmark as benchmark_module

    mock_mx = MagicMock()
    mock_load = MagicMock()

    benchmark_module.mx = mock_mx
    benchmark_module.load = mock_load

    # Mock __call__ attribute specifically if it's accessed that way
    mock_load.__call__ = MagicMock(return_value=["model", "tokenizer"])

    # cpu
    model, device = benchmark_module._load_mlx_model_and_device("model", "cpu")
    assert model == "model"
    assert device == "cpu"
    mock_mx.set_default_device.assert_called()

    # gpu
    model, device = benchmark_module._load_mlx_model_and_device("model", "gpu")
    assert device == "gpu"

    # not tuple
    mock_load.__call__ = MagicMock(return_value="model_only")
    model, device = benchmark_module._load_mlx_model_and_device("model", "gpu")
    assert model == "model_only"

    # mx set default device raises
    mock_mx.set_default_device.side_effect = ValueError("error")
    model, device = benchmark_module._load_mlx_model_and_device("model", "gpu")

    # mx missing set_default_device
    del mock_mx.set_default_device
    model, device = benchmark_module._load_mlx_model_and_device("model", "gpu")
    assert device == "gpu"
    assert model == "model_only"


def test_sync_and_eval():
    """Test sync and eval."""
    import gemma_4_sql.backends.mlx.benchmark as benchmark_module

    mock_mx = MagicMock()
    benchmark_module.mx = mock_mx

    benchmark_module._sync_and_eval(["t1", "t2"])
    mock_mx.eval.assert_called_with("t1", "t2")

    benchmark_module._sync_and_eval("t1")
    mock_mx.eval.assert_called_with("t1")

    benchmark_module.mx = None
    benchmark_module._sync_and_eval("t1")  # shouldn't raise


def test_get_peak_memory_mb():
    """Test get peak memory mb."""
    import gemma_4_sql.backends.mlx.benchmark as benchmark_module

    mock_mx = MagicMock()
    benchmark_module.mx = mock_mx

    # metal available
    mock_mx.metal.is_available.return_value = True
    mock_mx.metal.get_peak_memory.return_value = 1048576 * 10
    assert benchmark_module._get_peak_memory_mb() == 10.0

    # fallback to get_active_memory
    mock_mx.metal.get_peak_memory.return_value = 0
    mock_mx.metal.get_active_memory.return_value = 1048576 * 5
    assert benchmark_module._get_peak_memory_mb() == 5.0

    # missing get_active_memory but has mx.get_peak_memory
    del mock_mx.metal.get_active_memory
    mock_mx.get_peak_memory.return_value = 1048576 * 2
    assert benchmark_module._get_peak_memory_mb() == 2.0

    # fallback to get_peak_memory directly if metal is not available
    mock_mx.metal.is_available.return_value = False
    mock_mx.get_peak_memory.return_value = 1048576 * 2
    assert benchmark_module._get_peak_memory_mb() == 2.0

    # nothing available
    del mock_mx.get_peak_memory
    assert benchmark_module._get_peak_memory_mb() == 0.0

    # None
    benchmark_module.mx = None
    assert benchmark_module._get_peak_memory_mb() == 0.0


@patch("time.perf_counter")
def test_run_benchmark_pass(mock_perf):
    """Test run benchmark pass."""
    import gemma_4_sql.backends.mlx.benchmark as benchmark_module

    mock_mx = MagicMock()
    benchmark_module.mx = mock_mx

    mock_mx.metal.is_available.return_value = False
    mock_mx.get_peak_memory.return_value = 0
    mock_mx.zeros.return_value = MagicMock()

    mock_model = MagicMock()
    mock_model.return_value = "out"

    times = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    mock_perf.side_effect = times * 10

    tokens_sec, lat_ms, mem_mb = benchmark_module._run_benchmark_pass(model=mock_model, batch_size=1, num_runs=1, prompt_len=2, decode_tokens=2)

    assert tokens_sec > 0
    assert lat_ms > 0

    # metal reset peak memory
    mock_mx.metal.reset_peak_memory.side_effect = RuntimeError("error")
    benchmark_module._run_benchmark_pass(mock_model, 1, 1)

    # model is None
    del mock_mx.metal
    benchmark_module._run_benchmark_pass(None, 1, 1)


def test_benchmark_model():
    """Test benchmark_model."""
    import gemma_4_sql.backends.mlx.benchmark as benchmark_module

    benchmark_module.mx = MagicMock()
    benchmark_module.mx.metal.get_peak_memory.return_value = 100
    benchmark_module.load = MagicMock()

    with patch.object(benchmark_module, "run_benchmark_wrapper") as mock_wrapper:
        mock_wrapper.return_value = {"status": "ok"}
        res = benchmark_module.benchmark_model("model", "cpu", 1)
        assert res == {"status": "ok"}

        # Test benchmark fn
        fn = mock_wrapper.call_args[1]["benchmark_fn"]

        with patch.object(benchmark_module, "_load_mlx_model_and_device") as mock_load:
            mock_load.return_value = (MagicMock(), "cpu")
            with patch.object(benchmark_module, "_run_benchmark_pass") as mock_run:
                mock_run.return_value = (1.0, 2.0, 3.0)
                assert fn() == (1.0, 2.0, 3.0)


def test_get_peak_memory_mb_missing_metal_get_peak():
    """Test get_peak_memory_mb missing metal.get_peak_memory to hit branch."""
    from unittest.mock import MagicMock

    import gemma_4_sql.backends.mlx.benchmark as benchmark_module

    mock_mx = MagicMock()
    benchmark_module.mx = mock_mx
    mock_mx.metal.is_available.return_value = True
    del mock_mx.metal.get_peak_memory
    mock_mx.metal.get_active_memory.return_value = 1048576 * 3
    assert benchmark_module._get_peak_memory_mb() == 3.0


def test_mlx_benchmark_successful_imports():
    """Test mlx benchmark successful imports."""
    import sys
    from unittest.mock import MagicMock

    mock_mlx = MagicMock()
    mock_core = MagicMock()
    mock_lm = MagicMock()

    with patch.dict(
        sys.modules,
        {
            "mlx": mock_mlx,
            "mlx.core": mock_core,
            "mlx_lm": mock_lm,
        },
    ):
        if "gemma_4_sql.backends.mlx.benchmark" in sys.modules:
            del sys.modules["gemma_4_sql.backends.mlx.benchmark"]
        import gemma_4_sql.backends.mlx.benchmark as benchmark_module

        assert benchmark_module.mx is not None
        assert benchmark_module.load is not None

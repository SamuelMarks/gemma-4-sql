"""Module docstring."""

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_dependencies():
    """Docstring for mock_dependencies."""
    mock_jax = MagicMock()
    mock_jnp = MagicMock()
    mock_nnx = MagicMock()
    mock_flax = MagicMock()
    mock_flax.nnx = mock_nnx

    mock_jax.devices.side_effect = lambda x: [f"mock_{x}_device"] if x in ("gpu", "tpu", "cpu") else []

    # Return the original function for jit
    mock_nnx.jit.side_effect = lambda f: f

    mock_jnp.int32 = "int32"
    mock_jnp.bfloat16 = "bfloat16"

    # Model config
    mock_gemma4_config = MagicMock()
    mock_gemma4_config.gemma4_e2b.return_value = MagicMock()
    mock_gemma4_model = MagicMock()

    mock_gemma4_mod = MagicMock()
    mock_gemma4_mod.Gemma4Config = mock_gemma4_config
    mock_gemma4_mod.Gemma4ForCausalLM = mock_gemma4_model

    with patch.dict(
        sys.modules,
        {
            "jax": mock_jax,
            "jax.numpy": mock_jnp,
            "flax": mock_flax,
            "flax.nnx": mock_nnx,
            "gemma_4_sql.backends.jax.gemma4": mock_gemma4_mod,
        },
    ):
        yield mock_jax, mock_jnp, mock_nnx, mock_gemma4_config, mock_gemma4_model


def reload_module():
    """Docstring for reload_module."""
    import gemma_4_sql.backends.jax.benchmark as jax_benchmark

    importlib.reload(jax_benchmark)
    return jax_benchmark


def test_missing_dependencies():
    """Docstring for test_missing_dependencies."""
    with patch.dict(sys.modules, {"jax": None, "jax.numpy": None, "flax": None, "flax.nnx": None}):
        jax_benchmark = reload_module()
        with pytest.raises(Exception, match="JAX dependencies are missing."):
            jax_benchmark.benchmark_model("test", "cpu", 1)


def test_get_device():
    """Docstring for test_get_device."""
    jax_benchmark = reload_module()

    assert jax_benchmark._get_device("gpu") == "mock_gpu_device"
    assert jax_benchmark._get_device("tpu") == "mock_tpu_device"
    assert jax_benchmark._get_device("unknown") == "mock_cpu_device"

    def side_effect(hw):
        """Docstring for side_effect."""
        if hw in ("tpu", "gpu"):
            raise RuntimeError("Mock error")
        return [f"mock_{hw}_device"]

    jax_benchmark.jax.devices.side_effect = side_effect
    assert jax_benchmark._get_device("tpu") == "mock_cpu_device"


def test_run_benchmark_pass_prefill():
    """Docstring for test_run_benchmark_pass_prefill."""
    jax_benchmark = reload_module()

    mock_model = MagicMock()
    mock_device = MagicMock()

    mock_device.memory_stats.return_value = {"peak_bytes_in_use": 1024 * 1024 * 100}

    # We remove block_until_ready from jax to test the negative branch of hasattr
    del jax_benchmark.jax.block_until_ready

    tokens, latency, memory = jax_benchmark._run_benchmark_pass(model=mock_model, batch_size=2, num_runs=2, warmup_steps=1, mode="prefill", max_new_tokens=10, device=mock_device)

    assert isinstance(tokens, float)
    assert isinstance(latency, float)
    assert memory == 100.0


def test_run_benchmark_pass_generate():
    """Docstring for test_run_benchmark_pass_generate."""
    jax_benchmark = reload_module()
    del jax_benchmark.nnx.jit
    jax_benchmark = reload_module()

    mock_model = MagicMock()
    mock_model.return_value = MagicMock()

    jax_benchmark.jnp.argmax.return_value = MagicMock()
    # Mock sequence shape for arange
    mock_inputs = MagicMock()
    mock_inputs.shape = (1, 10)
    jax_benchmark.jnp.concatenate.return_value = mock_inputs
    jax_benchmark.jnp.arange.return_value = MagicMock()
    jax_benchmark.jnp.arange.return_value.__getitem__.return_value = MagicMock()

    mock_device = MagicMock()
    mock_device.memory_stats.side_effect = AttributeError("No memory stats")

    tokens, latency, memory = jax_benchmark._run_benchmark_pass(model=mock_model, batch_size=1, num_runs=1, warmup_steps=1, mode="generate", max_new_tokens=2, device=mock_device)

    assert isinstance(tokens, float)
    assert isinstance(latency, float)
    assert memory == 8192.0


def test_benchmark_model_execution():
    """Docstring for test_benchmark_model_execution."""
    jax_benchmark = reload_module()

    # We don't mock _run_benchmark_pass, we let it run with mocked jax
    with patch("gemma_4_sql.backends.jax.benchmark.run_benchmark_wrapper") as mock_wrapper:

        def side_effect(backend_name, model_name, hardware, batch_size, missing_deps, missing_status, benchmark_fn):
            """Docstring for side_effect."""
            return benchmark_fn()

        mock_wrapper.side_effect = side_effect

        # This will call _run() -> _run_benchmark_pass
        res = jax_benchmark.benchmark_model("test_model", "cpu", 2, dtype="float32", mode="generate", max_new_tokens=1, warmup_steps=1, num_runs=1)
        assert len(res) == 3

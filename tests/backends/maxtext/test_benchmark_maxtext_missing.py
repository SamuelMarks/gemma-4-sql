"""Module docstring."""

import builtins
import importlib
from unittest.mock import MagicMock, patch

import gemma_4_sql.backends.maxtext.benchmark as mod


def test_maxtext_benchmark_import_error():
    """Docstring for test_maxtext_benchmark_import_error."""
    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        """Docstring for mock_import."""
        if name == "maxtext" or name == "jax":
            raise ImportError("mock")
        return orig_import(name, *args, **kwargs)

    builtins.__import__ = mock_import
    try:
        importlib.reload(mod)
        assert mod.jax is None
    finally:
        builtins.__import__ = orig_import
        importlib.reload(mod)


def test_maxtext_benchmark_missing_branches():
    """Docstring for test_maxtext_benchmark_missing_branches."""
    mock_jax = MagicMock()
    del mock_jax.block_until_ready

    with patch.object(mod, "jax", mock_jax):
        with patch.object(mod, "jnp", MagicMock()):
            with patch("gemma_4_sql.backends.maxtext.benchmark.time.time", side_effect=[1, 2]):
                mod._run_benchmark_pass(model=MagicMock(), params={}, batch_size=1, num_runs=1, warmup_steps=1, mode="prefill", max_new_tokens=10, device="cpu")


def test_maxtext_benchmark_missing_170():
    """Docstring for test_maxtext_benchmark_missing_170."""
    from unittest.mock import MagicMock, patch

    import gemma_4_sql.backends.maxtext.benchmark as mod

    mock_jax = MagicMock()
    mock_jax.distributed.initialize.side_effect = RuntimeError("mock err")
    with patch.object(mod, "jax", mock_jax):
        with patch.object(mod, "_run_benchmark_pass", return_value=(1.0, 1.0, 1.0)):
            with patch("gemma_4_sql.backends.maxtext.benchmark.Gemma4Model", MagicMock()):
                mod.benchmark_model("test", "cpu", 1)

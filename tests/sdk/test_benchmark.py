"""Provide module docstring."""

import pytest

from gemma_4_sql.sdk.benchmark import benchmark

try:
    import keras
except ImportError:
    keras = None

try:
    import maxtext
except ImportError:
    maxtext = None


def test_benchmark_jax() -> object:
    """Initialize function test_benchmark_jax.

    Raises:
        AssertionError: Description.

    """
    res = benchmark("gemma-4", "gpu", 1, "jax")
    if not res["backend"] == "jax":
        raise AssertionError


@pytest.mark.skipif(keras is None, reason="Keras is not installed")
def test_benchmark_keras() -> object:
    """Initialize function test_benchmark_keras.

    Raises:
        AssertionError: Description.

    """
    res = benchmark("gemma-4", "gpu", 1, "keras")
    if not res["backend"] == "keras":
        raise AssertionError


@pytest.mark.skipif(maxtext is None, reason="MaxText is not installed")
def test_benchmark_maxtext() -> object:
    """Initialize function test_benchmark_maxtext.

    Raises:
        AssertionError: Description.

    """
    res = benchmark("gemma-4", "gpu", 1, "maxtext")
    if not res["backend"] == "maxtext":
        raise AssertionError


def test_benchmark_pytorch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test benchmark for pytorch missing deps."""
    import gemma_4_sql.backends.pytorch.benchmark as bm

    monkeypatch.setattr(bm, "torch", None)
    from gemma_4_sql.exceptions import DependencyMissingError

    with pytest.raises(DependencyMissingError):
        benchmark("gemma-4", "gpu", 1, "pytorch")


def test_benchmark_unknown() -> object:
    """Initialize function test_benchmark_unknown."""
    with pytest.raises(ValueError, match=r".*"):
        benchmark("gemma-4", "gpu", 1, "unknown")

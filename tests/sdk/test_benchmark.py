"""Provide module docstring."""

import pytest

from gemma_4_sql.sdk.benchmark import benchmark


def test_benchmark_jax() -> object:
    """Initialize function test_benchmark_jax."""
    res = benchmark("gemma-4", "gpu", 1, "jax")
    if not res["backend"] == "jax":
        raise AssertionError


def test_benchmark_keras() -> object:
    """Initialize function test_benchmark_keras."""
    res = benchmark("gemma-4", "gpu", 1, "keras")
    if not res["backend"] == "keras":
        raise AssertionError


def test_benchmark_maxtext() -> object:
    """Initialize function test_benchmark_maxtext."""
    res = benchmark("gemma-4", "gpu", 1, "maxtext")
    if not res["backend"] == "maxtext":
        raise AssertionError


def test_benchmark_pytorch() -> object:
    """Initialize function test_benchmark_pytorch."""
    res = benchmark("gemma-4", "gpu", 1, "pytorch")
    if not res["backend"] == "pytorch":
        raise AssertionError


def test_benchmark_unknown() -> object:
    """Initialize function test_benchmark_unknown."""
    with pytest.raises(ValueError, match=r".*"):
        benchmark("gemma-4", "gpu", 1, "unknown")

from gemma_4_sql.sdk.benchmark import benchmark
import pytest

def test_benchmark_jax():
    res = benchmark("gemma-4", "gpu", 1, "jax")
    assert res["backend"] == "jax"

def test_benchmark_keras():
    res = benchmark("gemma-4", "gpu", 1, "keras")
    assert res["backend"] == "keras"

def test_benchmark_maxtext():
    res = benchmark("gemma-4", "gpu", 1, "maxtext")
    assert res["backend"] == "maxtext"

def test_benchmark_pytorch():
    res = benchmark("gemma-4", "gpu", 1, "pytorch")
    assert res["backend"] == "pytorch"

def test_benchmark_unknown():
    with pytest.raises(ValueError):
        benchmark("gemma-4", "gpu", 1, "unknown")

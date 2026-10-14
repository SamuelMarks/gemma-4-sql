"""Behavioral tests for the SDK Benchmark module across backends."""

from __future__ import annotations

from typing import Any

import pytest

from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.sdk.benchmark import benchmark


def test_benchmark_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify benchmark executes or propagates expected status for JAX backend."""
    import gemma_4_sql.backends.jax.benchmark as jbm
    from gemma_4_sql.backends.jax.gemma4.config import ModelConfig

    tiny_cfg = ModelConfig(
        vocab_size=64,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
    )
    if jbm.Gemma4Config is not None:
        monkeypatch.setattr(jbm.Gemma4Config, "gemma4_e2b", staticmethod(lambda: tiny_cfg))
    if jbm.jax is not None:
        monkeypatch.setattr(jbm.jax, "jit", lambda fn, *_a, **_k: fn)

    res = benchmark("gemma-4", "cpu", 1, "jax", warmup_steps=1)
    assert res["backend"] == "jax"
    assert "tokens_per_sec" in res or "latency_ms" in res or "status" in res


def test_benchmark_keras_routing_and_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify benchmark forwards exact arguments to Keras backend and raises when TensorFlow is missing."""
    import gemma_4_sql.backends.keras as kb
    import gemma_4_sql.backends.keras.benchmark as kbm

    monkeypatch.setattr(kbm, "tf", None)
    with pytest.raises(DependencyMissingError):
        benchmark("gemma-4", "gpu", 1, "keras")

    captured: dict[str, Any] = {}

    def spy_bench(model_name: str, hardware: str, batch_size: int = 1, **kwargs: Any) -> dict[str, Any]:
        """Record forwarded benchmark arguments."""
        captured["model_name"] = model_name
        captured["hardware"] = hardware
        captured["batch_size"] = batch_size
        captured["kwargs"] = kwargs
        return {"backend": "keras", "model": model_name, "hardware": hardware, "batch_size": batch_size, "status": "completed"}

    monkeypatch.setattr(kbm, "keras", object())
    monkeypatch.setattr(kbm, "tf", object())
    monkeypatch.setattr(kbm, "benchmark_model", spy_bench)
    monkeypatch.setattr(kb, "benchmark_model", spy_bench)

    res = benchmark("gemma-4", "gpu", 4, "keras", dtype="float16", mode="decode")
    assert res["backend"] == "keras"
    assert captured["model_name"] == "gemma-4"
    assert captured["hardware"] == "gpu"
    assert captured["batch_size"] == 4
    assert captured["kwargs"]["dtype"] == "float16"
    assert captured["kwargs"]["mode"] == "decode"


def test_benchmark_maxtext_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify benchmark forwards arguments to the MaxText backend."""
    import gemma_4_sql.backends.maxtext.benchmark as mbm

    captured: dict[str, Any] = {}

    def spy_bench(model_name: str, hardware: str, batch_size: int = 1, **kwargs: Any) -> dict[str, Any]:
        """Record forwarded MaxText benchmark arguments."""
        captured["model_name"] = model_name
        captured["hardware"] = hardware
        captured["batch_size"] = batch_size
        captured["kwargs"] = kwargs
        return {"backend": "maxtext", "status": "completed"}

    monkeypatch.setattr(mbm, "benchmark_model", spy_bench)
    res = benchmark("gemma-4", "tpu", 2, "maxtext")
    assert res["backend"] == "maxtext"
    assert captured["model_name"] == "gemma-4"
    assert captured["hardware"] == "tpu"
    assert captured["batch_size"] == 2


def test_benchmark_pytorch_routing_and_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify benchmark forwards arguments to PyTorch backend and raises when torch is missing."""
    import gemma_4_sql.backends.pytorch.benchmark as bm

    captured: dict[str, Any] = {}

    def spy_bench(model_name: str, hardware: str, batch_size: int = 1, **kwargs: Any) -> dict[str, Any]:
        """Record forwarded PyTorch benchmark arguments."""
        captured["model_name"] = model_name
        captured["hardware"] = hardware
        captured["batch_size"] = batch_size
        return {"backend": "pytorch", "status": "completed"}

    monkeypatch.setattr(bm, "benchmark_model", spy_bench)
    res = benchmark("gemma-4", "cpu", 1, "pytorch")
    assert res["backend"] == "pytorch"
    assert captured["model_name"] == "gemma-4"
    assert captured["hardware"] == "cpu"

    monkeypatch.setattr(bm, "torch", None)
    monkeypatch.undo()
    monkeypatch.setattr(bm, "torch", None)
    with pytest.raises(DependencyMissingError):
        benchmark("gemma-4", "gpu", 1, "pytorch")


def test_benchmark_unknown() -> None:
    """Verify benchmark raises ValueError on an unknown backend identifier."""
    with pytest.raises(ValueError, match="Unknown backend: unknown"):
        benchmark("gemma-4", "gpu", 1, "unknown")

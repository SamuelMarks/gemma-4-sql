"""Unit tests for backend adapter delegation across all 5 backends."""

from __future__ import annotations

import sys
from typing import Any

import pytest

from gemma_4_sql.backends.jax.adapter import JaxBackend
from gemma_4_sql.backends.keras.adapter import KerasBackend
from gemma_4_sql.backends.maxtext.adapter import MaxTextBackend
from gemma_4_sql.backends.mlx.adapter import MLXBackend
from gemma_4_sql.backends.pytorch.adapter import PyTorchBackend
from gemma_4_sql.type_hints import DPOConfig, ETLConfig, TrainingConfig


@pytest.fixture
def dummy_train_config() -> TrainingConfig:
    """Return a synthetic TrainingConfig for adapter routing tests."""
    return TrainingConfig(model_name="dummy", dataset="dummy")


@pytest.fixture
def dummy_dpo_config() -> DPOConfig:
    """Return a synthetic DPOConfig for adapter routing tests."""
    return DPOConfig(model_name="dummy", dataset="dummy")


@pytest.fixture
def dummy_etl_config() -> ETLConfig:
    """Return a synthetic ETLConfig for adapter routing tests."""
    return ETLConfig(dataset_name="dummy", split="dummy")


@pytest.mark.parametrize("backend_cls", [JaxBackend, KerasBackend, MaxTextBackend, MLXBackend, PyTorchBackend])
def test_backend_adapter(
    backend_cls: type[Any],
    dummy_train_config: TrainingConfig,
    dummy_dpo_config: DPOConfig,
    dummy_etl_config: ETLConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify every backend adapter delegates all 10 protocol methods to its underlying modules."""
    adapter = backend_cls()
    adapter_mod = sys.modules[backend_cls.__module__]

    monkeypatch.setattr(adapter_mod.train, "train_model", lambda *_a, **_k: {"status": "trained"})
    monkeypatch.setattr(adapter_mod.dpo, "run_dpo", lambda *_a, **_k: {"status": "dpo_run"})
    monkeypatch.setattr(adapter_mod.etl, "build_dataloader", lambda *_a, **_k: {"status": "dataloader"})
    monkeypatch.setattr(adapter_mod.export, "export_model", lambda *_a, **_k: {"status": "exported"})
    monkeypatch.setattr(adapter_mod.logging, "log_metrics", lambda *_a, **_k: {"status": "logged"})
    monkeypatch.setattr(adapter_mod.peft, "apply_lora", lambda *_a, **_k: {"status": "lora_applied"})
    monkeypatch.setattr(adapter_mod.quantize, "quantize_model", lambda *_a, **_k: {"status": "quantized"})
    monkeypatch.setattr(adapter_mod.inference, "generate_sql", lambda *_a, **_k: {"status": "generated"})
    monkeypatch.setattr(adapter_mod.serve, "serve_model", lambda *_a, **_k: {"status": "served"})
    monkeypatch.setattr(adapter_mod.benchmark, "benchmark_model", lambda *_a, **_k: {"status": "benchmarked"})

    assert adapter.train_model(dummy_train_config) == {"status": "trained"}
    assert adapter.run_dpo(dummy_dpo_config) == {"status": "dpo_run"}
    assert adapter.build_dataloader(dummy_etl_config) == {"status": "dataloader"}
    assert adapter.export_model("model", "path") == {"status": "exported"}
    assert adapter.log_metrics({"loss": 1.0}, 1) == {"status": "logged"}
    assert adapter.apply_lora("model", ["q_proj"]) == {"status": "lora_applied"}
    assert adapter.quantize_model("model") == {"status": "quantized"}
    assert adapter.generate_sql("model", "prompt") == {"status": "generated"}
    assert adapter.serve_model("model") == {"status": "served"}
    assert adapter.benchmark_model("model") == {"status": "benchmarked"}

    aliased = PyTorchBackend(backend_alias="pytorch_native")
    captured_kw: dict[str, Any] = {}
    monkeypatch.setattr(
        sys.modules[PyTorchBackend.__module__].inference,
        "generate_sql",
        lambda *_a, **kw: captured_kw.update(kw) or {"status": "aliased"},
    )
    assert aliased.generate_sql("model", "prompt") == {"status": "aliased"}
    assert captured_kw["backend_alias"] == "pytorch_native"

import pytest

from gemma_4_sql.backends.jax.adapter import JaxBackend
from gemma_4_sql.backends.keras.adapter import KerasBackend
from gemma_4_sql.backends.maxtext.adapter import MaxTextBackend
from gemma_4_sql.backends.mlx.adapter import MLXBackend
from gemma_4_sql.backends.pytorch.adapter import PyTorchBackend
from gemma_4_sql.type_hints import DPOConfig, ETLConfig, TrainingConfig


@pytest.fixture
def dummy_train_config():
    return TrainingConfig(model_name="dummy", dataset="dummy")


@pytest.fixture
def dummy_dpo_config():
    return DPOConfig(model_name="dummy", dataset="dummy")


@pytest.fixture
def dummy_etl_config():
    return ETLConfig(dataset_name="dummy", split="dummy")


@pytest.mark.parametrize("backend_cls", [JaxBackend, KerasBackend, MaxTextBackend, MLXBackend, PyTorchBackend])
def test_backend_adapter(backend_cls, dummy_train_config, dummy_dpo_config, dummy_etl_config, monkeypatch):
    adapter = backend_cls()

    import sys

    # Mock the functions directly on the adapter's module references
    adapter_mod = sys.modules[backend_cls.__module__]

    monkeypatch.setattr(adapter_mod.train, "train_model", lambda *args, **kwargs: {"status": "trained"})
    monkeypatch.setattr(adapter_mod.dpo, "run_dpo", lambda *args, **kwargs: {"status": "dpo_run"})
    monkeypatch.setattr(adapter_mod.etl, "build_dataloader", lambda *args, **kwargs: {"status": "dataloader"})
    monkeypatch.setattr(adapter_mod.export, "export_model", lambda *args, **kwargs: {"status": "exported"})
    monkeypatch.setattr(adapter_mod.logging, "log_metrics", lambda *args, **kwargs: {"status": "logged"})
    monkeypatch.setattr(adapter_mod.peft, "apply_lora", lambda *args, **kwargs: {"status": "lora_applied"})
    monkeypatch.setattr(adapter_mod.quantize, "quantize_model", lambda *args, **kwargs: {"status": "quantized"})
    monkeypatch.setattr(adapter_mod.inference, "generate_sql", lambda *args, **kwargs: {"status": "generated"})
    monkeypatch.setattr(adapter_mod.serve, "serve_model", lambda *args, **kwargs: {"status": "served"})
    monkeypatch.setattr(adapter_mod.benchmark, "benchmark_model", lambda *args, **kwargs: {"status": "benchmarked"})

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

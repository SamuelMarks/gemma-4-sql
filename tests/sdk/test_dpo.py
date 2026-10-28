"""Behavioral unit tests for the DPO SDK module."""

from __future__ import annotations

import pytest

import gemma_4_sql.sdk.dpo as dpo_sdk
from gemma_4_sql.exceptions import ConfigurationError, DependencyMissingError
from gemma_4_sql.sdk.dpo import run_dpo
from tests.utils.test_utils import RecordingBackendDouble


def test_run_dpo_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `run_dpo` constructs a `DPOConfig` and forwards `**kwargs` to the JAX backend."""
    jax_double = RecordingBackendDouble("jax")
    monkeypatch.setattr(dpo_sdk, "get_backend", lambda _b: jax_double)  # pyright: ignore[reportUnknownLambdaType]

    res = run_dpo("model1", "data1", "jax", 0.1, epochs=3, learning_rate=2e-5, batch_size=4, custom_flag="enabled")
    assert res["backend"] == "jax"
    assert res["action"] == "dpo"
    assert res["model"] == "model1"
    assert res["dataset"] == "data1"
    assert res["beta"] == 0.1
    assert len(jax_double.dpo_configs) == 1
    assert jax_double.dpo_configs[0].epochs == 3
    assert jax_double.dpo_configs[0].learning_rate == 2e-5
    assert jax_double.dpo_configs[0].batch_size == 4
    assert jax_double.dpo_configs[0].extra_kwargs == {"custom_flag": "enabled"}
    assert jax_double.dpo_kwargs == [{}]


def test_run_dpo_pytorch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `run_dpo` executes a real 1-step PyTorch DPO update on a tiny Gemma 4 model."""
    import torch

    import gemma_4_sql.backends.pytorch.dpo as pt_dpo

    monkeypatch.setattr(
        pt_dpo,
        "build_dataloader",
        lambda _cfg: {  # pyright: ignore[reportUnknownLambdaType]
            "loader": [
                {
                    "chosen_inputs": torch.tensor([[1, 2, 3]], dtype=torch.long),
                    "chosen_labels": torch.tensor([[2, 3, 4]], dtype=torch.long),
                    "rejected_inputs": torch.tensor([[1, 5, 6]], dtype=torch.long),
                    "rejected_labels": torch.tensor([[5, 6, 7]], dtype=torch.long),
                }
            ]
        },
    )
    res = run_dpo(
        model_name="gemma-4-tiny",
        dataset="synthetic-dpo",
        beta=0.2,
        epochs=1,
        learning_rate=1e-4,
        batch_size=1,
        backend="pytorch_native",
    )

    assert res["backend"] == "pytorch_native"
    assert res["action"] == "dpo"
    assert res["model"] == "gemma-4-tiny"
    assert res["dataset"] == "synthetic-dpo"
    assert res["beta"] == 0.2
    assert res["status"] == "completed"
    assert res["final_loss"] > 0.0


def test_run_dpo_keras(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `run_dpo` with the Keras backend propagates `DependencyMissingError` and executes `kdpo.run_dpo`."""
    import gemma_4_sql.backends.keras.dpo as kdpo

    monkeypatch.setattr(kdpo, "tf", None)
    with pytest.raises(DependencyMissingError):
        run_dpo(model_name="model3", dataset="data3", backend="keras", beta=0.3)

    class DummyModel:
        """Minimal callable model stub."""

        def __call__(self, x: list[list[int]]) -> list[list[int]]:
            """Return input batch unchanged."""
            return x

    class DummyGemmaCausalLM:
        """Stub for GemmaCausalLM.from_preset."""

        @staticmethod
        def from_preset(_name: str) -> DummyModel:
            """Return a dummy callable model."""
            mod = DummyModel()
            assert mod([[1]]) == [[1]]
            return mod

    class DummyKerasMod:
        """Stub for keras module with AdamW."""

        class optimizers:
            """Stub optimizers namespace."""

            @staticmethod
            def AdamW(**_kw: float | str) -> str:
                """Return optimizer stub."""
                return "adamw"

    class DummyTFModule:
        """Minimal TensorFlow sentinel."""

    monkeypatch.setattr(kdpo, "keras", DummyKerasMod)
    monkeypatch.setattr(kdpo, "tf", DummyTFModule())
    monkeypatch.setattr(kdpo, "GemmaCausalLM", DummyGemmaCausalLM)
    monkeypatch.setattr(kdpo, "_get_train_step_fn", lambda *_a: lambda _b: 0.15)  # pyright: ignore[reportUnknownLambdaType]
    monkeypatch.setattr(
        kdpo,
        "build_dataloader",
        lambda _cfg: {  # pyright: ignore[reportUnknownLambdaType]
            "loader": [
                {
                    "chosen_inputs": [[1, 2]],
                    "chosen_labels": [[1, 2]],
                    "rejected_inputs": [[1, 3]],
                    "rejected_labels": [[1, 3]],
                }
            ]
        },
    )

    res = run_dpo(model_name="model3", dataset="data3", backend="keras", beta=0.3)
    assert res["backend"] == "keras"
    assert res["action"] == "dpo"
    assert res["model"] == "model3"
    assert res["beta"] == 0.3
    assert res["status"] == "completed"
    assert res["final_loss"] == pytest.approx(0.15)


def test_run_dpo_jax_real_tiny_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `run_dpo` executes a real 1-step JAX DPO preference update on a tiny Gemma 4 model."""
    import jax.numpy as jnp

    import gemma_4_sql.backends.jax.dpo as jax_dpo

    monkeypatch.setattr(
        jax_dpo,
        "build_dataloader",
        lambda _cfg: {  # pyright: ignore[reportUnknownLambdaType]
            "loader": [
                {
                    "chosen_inputs": jnp.array([[1, 2, 3]], dtype=jnp.int32),
                    "chosen_labels": jnp.array([[2, 3, 4]], dtype=jnp.int32),
                    "rejected_inputs": jnp.array([[1, 5, 6]], dtype=jnp.int32),
                    "rejected_labels": jnp.array([[5, 6, 7]], dtype=jnp.int32),
                }
            ]
        },
    )
    res = run_dpo("gemma-4-tiny", "synthetic-dpo", "jax", beta=0.1, epochs=1, learning_rate=1e-4, batch_size=1)
    assert res["backend"] == "jax"
    assert res["status"] == "completed"
    assert res["final_loss"] > 0.0


def test_run_dpo_maxtext() -> None:
    """Verify `run_dpo` with MaxText raises `DependencyMissingError` when MaxText is uninstalled."""
    with pytest.raises(DependencyMissingError):
        run_dpo("model4", "data4", "maxtext", 0.4)


def test_run_dpo_invalid() -> None:
    """Verify `run_dpo` raises `ConfigurationError` (a `ValueError` subclass) on an unknown backend."""
    with pytest.raises(ConfigurationError, match="Unknown backend: unknown"):
        run_dpo("model", "data", "unknown")

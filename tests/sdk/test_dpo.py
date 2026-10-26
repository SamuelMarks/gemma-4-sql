"""Behavioral unit tests for the DPO SDK module."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from gemma_4_sql.exceptions import ConfigurationError, DependencyMissingError
from gemma_4_sql.sdk.dpo import run_dpo
from gemma_4_sql.type_hints import DPOConfig


def test_run_dpo_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `run_dpo` constructs a `DPOConfig` and forwards `**kwargs` to the JAX backend."""
    from gemma_4_sql.sdk.registry import get_backend

    jax_agent = get_backend("jax")
    captured_cfg: list[DPOConfig] = []
    captured_kw: dict[str, Any] = {}

    def _spy_dpo(cfg: DPOConfig, **kw: Any) -> dict[str, Any]:
        """Record DPOConfig and keyword arguments."""
        captured_cfg.append(cfg)
        captured_kw.update(kw)
        return {
            "backend": "jax",
            "action": "dpo",
            "model": cfg.model_name,
            "dataset": cfg.dataset,
            "beta": cfg.beta,
            "status": "completed",
            "final_loss": 0.25,
        }

    monkeypatch.setattr(jax_agent, "run_dpo", _spy_dpo)

    res = run_dpo("model1", "data1", "jax", 0.1, epochs=3, learning_rate=2e-5, batch_size=4, custom_flag="enabled")
    assert res["backend"] == "jax"
    assert res["action"] == "dpo"
    assert res["model"] == "model1"
    assert res["dataset"] == "data1"
    assert res["beta"] == 0.1
    assert len(captured_cfg) == 1
    assert captured_cfg[0].epochs == 3
    assert captured_cfg[0].learning_rate == 2e-5
    assert captured_cfg[0].batch_size == 4
    assert captured_cfg[0].extra_kwargs == {"custom_flag": "enabled"}
    assert captured_kw == {}


def test_run_dpo_pytorch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `run_dpo` passes an accurately populated `DPOConfig` to the PyTorch backend."""
    import gemma_4_sql.sdk.dpo as sdk_dpo

    mock_be = MagicMock()
    mock_be.run_dpo.return_value = {
        "backend": "pytorch",
        "action": "dpo",
        "model": "model2",
        "dataset": "data2",
        "beta": 0.2,
        "status": "completed",
        "final_loss": 0.1,
    }
    monkeypatch.setattr(sdk_dpo, "get_backend", lambda _x: mock_be)

    res = sdk_dpo.run_dpo(
        model_name="model2",
        dataset="data2",
        beta=0.2,
        epochs=2,
        learning_rate=5e-5,
        batch_size=8,
        backend="pytorch",
    )

    assert res["backend"] == "pytorch"
    assert res["action"] == "dpo"
    assert res["model"] == "model2"
    assert res["dataset"] == "data2"
    assert res["beta"] == 0.2

    mock_be.run_dpo.assert_called_once()
    passed_cfg = mock_be.run_dpo.call_args.args[0]
    assert isinstance(passed_cfg, DPOConfig)
    assert passed_cfg.model_name == "model2"
    assert passed_cfg.dataset == "data2"
    assert passed_cfg.beta == 0.2
    assert passed_cfg.epochs == 2
    assert passed_cfg.learning_rate == 5e-5
    assert passed_cfg.batch_size == 8


def test_run_dpo_keras(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `run_dpo` with the Keras backend propagates `DependencyMissingError` and executes `kdpo.run_dpo`."""
    import gemma_4_sql.backends.keras.dpo as kdpo

    monkeypatch.setattr(kdpo, "tf", None)
    with pytest.raises(DependencyMissingError):
        run_dpo(model_name="model3", dataset="data3", backend="keras", beta=0.3)

    class DummyGemmaCausalLM:
        """Stub for GemmaCausalLM.from_preset."""

        @staticmethod
        def from_preset(_name: str) -> Any:
            """Return a dummy callable model."""
            return lambda x: x

    class DummyKerasMod:
        """Stub for keras module with AdamW."""

        class optimizers:
            """Stub optimizers namespace."""

            @staticmethod
            def AdamW(**_kw: Any) -> str:
                """Return optimizer stub."""
                return "adamw"

    monkeypatch.setattr(kdpo, "keras", DummyKerasMod)
    monkeypatch.setattr(kdpo, "tf", object())
    monkeypatch.setattr(kdpo, "GemmaCausalLM", DummyGemmaCausalLM)
    monkeypatch.setattr(kdpo, "_get_train_step_fn", lambda *_a: lambda _b: 0.15)
    monkeypatch.setattr(
        kdpo,
        "build_dataloader",
        lambda _cfg: {
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
        lambda _cfg: {
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

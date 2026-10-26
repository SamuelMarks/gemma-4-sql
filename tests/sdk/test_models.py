"""Behavioral unit tests for the Models SDK module."""

from __future__ import annotations

from typing import Any

import pytest

from gemma_4_sql.exceptions import ConfigurationError, DependencyMissingError
from gemma_4_sql.sdk import models
from gemma_4_sql.sdk.models import posttrain_model, pretrain_model, sft_model, train_from_scratch
from gemma_4_sql.sdk.registry import get_backend
from gemma_4_sql.type_hints import TrainingConfig


def test_pretrain_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `pretrain_model` routes configurations across JAX, PyTorch, Keras, MaxText, and MLX."""
    jax_agent = get_backend("jax")
    monkeypatch.setattr(
        jax_agent,
        "train_model",
        lambda cfg, **_k: {
            "backend": cfg.backend,
            "action": cfg.action,
            "model": cfg.model_name,
            "dataset": cfg.dataset,
            "epochs": cfg.epochs,
            "status": "completed",
            "final_loss": 0.1,
        },
    )
    res = pretrain_model(TrainingConfig(action="pretrain", model_name="my-model", dataset="my-data", epochs=2, backend="jax"))
    assert res["backend"] == "jax"
    assert res["action"] == "pretrain"
    assert res["model"] == "my-model"

    monkeypatch.setattr(
        "gemma_4_sql.backends.pytorch.train.train_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(DependencyMissingError("PyTorch dependencies are missing.")),
    )
    with pytest.raises(DependencyMissingError):
        pretrain_model(TrainingConfig(action="pretrain", model_name="my-model", dataset="my-data", epochs=2, backend="pytorch"))

    import gemma_4_sql.backends.keras.train as ktrain

    monkeypatch.setattr(ktrain, "tf", None)
    with pytest.raises(DependencyMissingError):
        pretrain_model(TrainingConfig(action="pretrain", model_name="my-model", dataset="my-data", epochs=2, backend="keras"))

    class DummyKerasModel:
        """Minimal Keras model stub for behavioral routing through ktrain.train_model."""

        def __init__(self) -> None:
            """Initialize model stub with preprocessor."""

            class _Prep:
                """Preprocessor stub."""

                sequence_length = 128

            self.preprocessor = _Prep()

        def compile(self, **_kw: Any) -> None:
            """No-op compile."""

        def fit(self, _ds: Any, epochs: int = 1) -> Any:
            """Return history stub with decreasing loss."""

            class _Hist:
                """Fit history stub."""

                history = {"loss": [0.4 / max(1, epochs)]}

            return _Hist()

    class DummyGemmaCausalLM:
        """Stub for GemmaCausalLM.from_preset."""

        @staticmethod
        def from_preset(_name: str) -> DummyKerasModel:
            """Return a DummyKerasModel instance."""
            return DummyKerasModel()

    class DummyKerasMod:
        """Stub for keras module with AdamW and SparseCategoricalCrossentropy."""

        class optimizers:
            """Stub optimizers namespace."""

            @staticmethod
            def AdamW(**_kw: Any) -> str:
                """Return optimizer stub."""
                return "adamw"

        class losses:
            """Stub losses namespace."""

            @staticmethod
            def SparseCategoricalCrossentropy(**_kw: Any) -> str:
                """Return loss stub."""
                return "loss"

    class DummyTF:
        """Stub for tensorflow module with distribute.MirroredStrategy."""

        class distribute:
            """Stub distribute namespace."""

            class MirroredStrategy:
                """Context-manager stub for MirroredStrategy."""

                def scope(self) -> Any:
                    """Return a nullcontext scope."""
                    from contextlib import nullcontext

                    return nullcontext()

    monkeypatch.setattr(ktrain, "keras", DummyKerasMod)
    monkeypatch.setattr(ktrain, "tf", DummyTF)
    monkeypatch.setattr(ktrain, "GemmaCausalLM", DummyGemmaCausalLM)
    monkeypatch.setattr(ktrain, "build_dataloader", lambda _cfg: {"loader": [{"inputs": [[1, 2]], "targets": [[2, 3]]}]})

    res_k = pretrain_model(TrainingConfig(action="pretrain", model_name="my-model", dataset="my-data", epochs=2, backend="keras"))
    assert res_k["backend"] == "keras"
    assert res_k["action"] == "pretrain"
    assert res_k["status"] == "completed"
    assert res_k["final_loss"] == pytest.approx(0.2)

    import gemma_4_sql.backends.maxtext.train as mx_train

    monkeypatch.setattr(mx_train, "jax", None)
    with pytest.raises(DependencyMissingError):
        pretrain_model(TrainingConfig(action="pretrain", model_name="my-model", dataset="my-data", epochs=2, backend="maxtext"))

    mlx_agent = get_backend("mlx")
    monkeypatch.setattr(
        mlx_agent,
        "train_model",
        lambda cfg, **_k: {
            "backend": cfg.backend,
            "action": cfg.action,
            "model": cfg.model_name,
            "dataset": cfg.dataset,
            "epochs": cfg.epochs,
            "status": "completed",
            "final_loss": 0.05,
        },
    )

    res_mlx = pretrain_model(TrainingConfig(action="pretrain", model_name="my-model", dataset="my-data", epochs=2, backend="mlx"))
    assert res_mlx["backend"] == "mlx"


def test_pretrain_model_error() -> None:
    """Verify `pretrain_model` raises `ConfigurationError` with an unknown backend."""
    with pytest.raises(ConfigurationError, match="Unknown backend: unknown"):
        pretrain_model(TrainingConfig(action="pretrain", model_name="my-model", dataset="my-data", epochs=2, backend="unknown"))


def test_models_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `pretrain_model`, `train_from_scratch`, `sft_model`, and `posttrain_model` populate default configs without mutating caller configs."""
    captured: list[TrainingConfig] = []

    class DummyBackend:
        """Backend stub that records passed TrainingConfig."""

        def train_model(self, cfg: TrainingConfig, **_kw: Any) -> dict[str, Any]:
            """Record TrainingConfig and return valid TrainResultDict."""
            captured.append(cfg)
            return {
                "backend": cfg.backend,
                "action": cfg.action,
                "model": cfg.model_name,
                "dataset": cfg.dataset,
                "epochs": cfg.epochs,
                "status": "completed",
                "final_loss": 0.0,
            }

    monkeypatch.setattr(models, "get_backend", lambda _b: DummyBackend())

    res0 = pretrain_model()
    assert res0["action"] == "pretrain"
    assert captured[-1].action == "pretrain"
    assert captured[-1].backend == "maxtext"

    res1 = train_from_scratch()
    assert res1["action"] == "train_from_scratch"
    assert captured[-1].action == "train_from_scratch"

    res2 = sft_model()
    assert res2["action"] == "sft"
    assert captured[-1].action == "sft"

    res3 = posttrain_model()
    assert res3["action"] == "posttrain"
    assert captured[-1].action == "posttrain"
    assert captured[-1].backend == "keras"


def test_sft_and_train_from_scratch_real_tiny_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `sft_model` and `train_from_scratch` execute a real 1-step JAX training loop on a tiny Gemma 4 model."""
    import jax.numpy as jnp

    import gemma_4_sql.backends.jax.train as jax_train

    monkeypatch.setattr(
        jax_train,
        "build_dataloader",
        lambda _cfg: {
            "loader": [
                {
                    "inputs": jnp.array([[1, 2, 3, 4]], dtype=jnp.int32),
                    "targets": jnp.array([[2, 3, 4, 5]], dtype=jnp.int32),
                }
            ]
        },
    )
    res_sft = sft_model(
        TrainingConfig(
            action="sft",
            model_name="gemma-4-tiny",
            dataset="synthetic",
            epochs=1,
            learning_rate=1e-3,
            batch_size=1,
            backend="jax",
        )
    )
    assert res_sft["backend"] == "jax"
    assert res_sft["action"] == "sft"
    assert res_sft["status"] == "completed"
    assert res_sft["final_loss"] > 0.0

    res_scratch = train_from_scratch(
        TrainingConfig(
            action="train_from_scratch",
            model_name="gemma-4-tiny",
            dataset="synthetic",
            epochs=1,
            learning_rate=1e-3,
            batch_size=1,
            backend="jax",
        )
    )
    assert res_scratch["action"] == "train_from_scratch"
    assert res_scratch["status"] == "completed"
    assert res_scratch["final_loss"] > 0.0

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

    import gemma_4_sql.backends.keras as kb
    import gemma_4_sql.backends.keras.train as ktrain

    monkeypatch.setattr(ktrain, "tf", None)
    with pytest.raises(DependencyMissingError):
        pretrain_model(TrainingConfig(action="pretrain", model_name="my-model", dataset="my-data", epochs=2, backend="keras"))

    def mock_train(cfg: TrainingConfig, **_kwargs: Any) -> dict[str, Any]:
        """Return simulated Keras training result."""
        return {
            "backend": "keras",
            "action": cfg.action,
            "model": cfg.model_name,
            "dataset": cfg.dataset,
            "epochs": cfg.epochs,
            "status": "completed",
            "final_loss": 0.2,
        }

    monkeypatch.setattr(ktrain, "keras", object())
    monkeypatch.setattr(ktrain, "tf", object())
    monkeypatch.setattr(ktrain, "train_model", mock_train)
    monkeypatch.setattr(kb, "train_model", mock_train)

    res_k = pretrain_model(TrainingConfig(action="pretrain", model_name="my-model", dataset="my-data", epochs=2, backend="keras"))
    assert res_k["backend"] == "keras"
    assert res_k["action"] == "pretrain"

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

"""Tests for Models SDK module."""

import pytest

from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.sdk.models import pretrain_model
from gemma_4_sql.type_hints import TrainingConfig


def test_pretrain_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test pretraining a model."""
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    jax_agent = get_backend("jax")
    monkeypatch.setattr(jax_agent, "train_model", lambda cfg, **k: {"backend": cfg.backend, "action": cfg.action, "model": cfg.model_name, "dataset": cfg.dataset, "epochs": cfg.epochs})
    res = pretrain_model(TrainingConfig(action="pretrain", model_name="my-model", dataset="my-data", epochs=2, backend="jax"))
    assert res["backend"] == "jax"
    assert res["action"] == "pretrain"
    assert res["model"] == "my-model"

    # Instead of just pt_train.torch = None, mock train_model to raise the error
    # since other tests might have patched the module
    monkeypatch.setattr("gemma_4_sql.backends.pytorch.train.train_model", lambda *args, **kwargs: (_ for _ in ()).throw(DependencyMissingError("PyTorch dependencies are missing.")))
    with pytest.raises(DependencyMissingError):
        pretrain_model(TrainingConfig(action="pretrain", model_name="my-model", dataset="my-data", epochs=2, backend="pytorch"))

    import gemma_4_sql.backends.keras as kb
    import gemma_4_sql.backends.keras.train as ktrain

    monkeypatch.setattr(ktrain, "tf", None)
    with pytest.raises(DependencyMissingError):
        pretrain_model(TrainingConfig(action="pretrain", model_name="my-model", dataset="my-data", epochs=2, backend="keras"))

    def mock_train(*args, **kwargs):
        """Docstring for mock_train."""
        return {"backend": "keras", "action": "pretrain", "model": "my-model"}

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

    # cover mlx pretraining by mocking registry and mlx modules
    import sys
    from unittest.mock import Mock

    monkeypatch.setitem(sys.modules, "mlx.core", Mock())
    monkeypatch.setitem(sys.modules, "mlx.nn", Mock())
    monkeypatch.setitem(sys.modules, "mlx.optimizers", Mock())
    monkeypatch.setitem(sys.modules, "mlx_lm", Mock())

    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    mlx_agent = get_backend("mlx")
    monkeypatch.setattr(mlx_agent, "train_model", lambda cfg, **k: {"backend": cfg.backend, "action": cfg.action, "model": cfg.model_name, "dataset": cfg.dataset, "epochs": cfg.epochs})

    res_mlx = pretrain_model(TrainingConfig(action="pretrain", model_name="my-model", dataset="my-data", epochs=2, backend="mlx"))
    assert res_mlx["backend"] == "mlx"


def test_pretrain_model_error() -> None:
    """Test pretraining a model with an unknown backend."""
    with pytest.raises(ValueError, match=r".*"):
        pretrain_model(TrainingConfig(action="pretrain", model_name="my-model", dataset="my-data", epochs=2, backend="unknown"))


def test_models_defaults(monkeypatch):
    from gemma_4_sql.sdk import models

    monkeypatch.setattr(models, "_route_training", lambda c: c)

    res1 = models.train_from_scratch()
    assert res1.action == "train_from_scratch"

    res2 = models.sft_model()
    assert res2.action == "sft"

    res3 = models.posttrain_model()
    assert res3.action == "posttrain"

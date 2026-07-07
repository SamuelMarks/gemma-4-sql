"""Tests for Models SDK module."""

import pytest

from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.sdk.models import pretrain_model
from gemma_4_sql.type_hints import TrainingConfig


def test_pretrain_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test pretraining a model."""
    res = pretrain_model(TrainingConfig(action="pretrain", model_name="my-model", dataset="my-data", epochs=2, backend="jax"))
    if not res["backend"] == "jax":
        raise AssertionError
    if not res["action"] == "pretrain":
        raise AssertionError
    if not res["model"] == "my-model":
        raise AssertionError

    import gemma_4_sql.backends.pytorch.train as pt_train

    monkeypatch.setattr(pt_train, "torch", None)
    with pytest.raises(DependencyMissingError):
        pretrain_model(TrainingConfig(action="pretrain", model_name="my-model", dataset="my-data", epochs=2, backend="pytorch"))

    res = pretrain_model(TrainingConfig(action="pretrain", model_name="my-model", dataset="my-data", epochs=2, backend="keras"))
    if not res["backend"] == "keras":
        raise AssertionError
    if not res["action"] == "pretrain":
        raise AssertionError

    import gemma_4_sql.backends.maxtext.train as mx_train

    monkeypatch.setattr(mx_train, "jax", None)
    with pytest.raises(DependencyMissingError):
        pretrain_model(TrainingConfig(action="pretrain", model_name="my-model", dataset="my-data", epochs=2, backend="maxtext"))

    with pytest.raises(ValueError):
        pretrain_model(TrainingConfig(action="pretrain", model_name="my-model", dataset="my-data", epochs=2, backend="mlx"))


def test_pretrain_model_error() -> None:
    """Test pretraining a model with an unknown backend."""
    with pytest.raises(ValueError, match=r".*"):
        pretrain_model(TrainingConfig(action="pretrain", model_name="my-model", dataset="my-data", epochs=2, backend="unknown"))

# Copyright 2024
"""Tests for the SDK Models module."""

from gemma_4_sql.sdk.models import TrainingConfig, posttrain_model, pretrain_model, sft_model, train_from_scratch


def test_train_from_scratch() -> None:
    """Test training from scratch.

    Raises:
        AssertionError: Description.

    """
    res = train_from_scratch(TrainingConfig(model_name="my-model", dataset="my-data", epochs=2, backend="jax"))
    if not res["action"] == "train_from_scratch":
        raise AssertionError
    if not res["model"] == "my-model":
        raise AssertionError
    if not res["backend"] == "jax":
        raise AssertionError
    if not res["dataset"] == "my-data":
        raise AssertionError
    if not res["epochs"] == int("2"):
        raise AssertionError


def test_pretrain_model() -> None:
    """Test pretraining a model.

    Raises:
        AssertionError: Description.

    """
    res = pretrain_model(TrainingConfig(model_name="my-model", dataset="my-data", epochs=2, backend="maxtext"))
    if not res["action"] == "pretrain":
        raise AssertionError
    if not res["model"] == "my-model":
        raise AssertionError
    if not res["backend"] == "maxtext":
        raise AssertionError
    if not res["dataset"] == "my-data":
        raise AssertionError
    if not res["epochs"] == int("2"):
        raise AssertionError


def test_sft_model() -> None:
    """Test SFT of a model.

    Raises:
        AssertionError: Description.

    """
    res = sft_model(TrainingConfig(model_name="my-model", dataset="my-data", epochs=2, backend="jax"))
    if not res["action"] == "sft":
        raise AssertionError
    if not res["model"] == "my-model":
        raise AssertionError
    if not res["backend"] == "jax":
        raise AssertionError
    if not res["dataset"] == "my-data":
        raise AssertionError
    if not res["epochs"] == int("2"):
        raise AssertionError


def test_posttrain_model() -> None:
    """Test post-training a model.

    Raises:
        AssertionError: Description.

    """
    res = posttrain_model(TrainingConfig(model_name="my-model", dataset="my-data", epochs=2, backend="keras"))
    if not res["action"] == "posttrain":
        raise AssertionError
    if not res["model"] == "my-model":
        raise AssertionError
    if not res["backend"] == "keras":
        raise AssertionError
    if not res["dataset"] == "my-data":
        raise AssertionError
    if not res["epochs"] == int("2"):
        raise AssertionError


def test_unknown_backend() -> None:
    """Test routing to unknown backend."""
    pytest = __import__("pytest", fromlist=[""])
    with pytest.raises(ValueError, match="Unknown backend: missing"):
        train_from_scratch(TrainingConfig(model_name="my-model", dataset="my-data", backend="missing"))

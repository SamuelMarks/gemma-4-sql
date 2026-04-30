"""
Tests for the SDK Models module.
"""

from gemma_4_sql.sdk.models import (
    posttrain_model,
    pretrain_model,
    sft_model,
    train_from_scratch,
)


def test_train_from_scratch() -> None:
    """Test training from scratch."""
    res = train_from_scratch("my-model", "my-data", epochs=2, backend="jax")
    assert res["action"] == "train_from_scratch"
    assert res["model"] == "my-model"
    assert res["backend"] == "jax"
    assert res["status"] == "mocked_missing_jax"
    assert res["dataset"] == "my-data"
    assert res["epochs"] == 2


def test_pretrain_model() -> None:
    """Test pretraining a model."""
    res = pretrain_model("my-model", "my-data", epochs=2, backend="maxtext")
    assert res["action"] == "pretrain"
    assert res["model"] == "my-model"
    assert res["backend"] == "maxtext"
    assert res["status"] == "mocked_missing_maxtext"
    assert res["dataset"] == "my-data"
    assert res["epochs"] == 2


def test_sft_model() -> None:
    """Test SFT of a model."""
    res = sft_model("my-model", "my-data", epochs=2, backend="jax")
    assert res["action"] == "sft"
    assert res["model"] == "my-model"
    assert res["backend"] == "jax"
    assert res["status"] == "mocked_missing_jax"
    assert res["dataset"] == "my-data"
    assert res["epochs"] == 2


def test_posttrain_model() -> None:
    """Test post-training a model."""
    res = posttrain_model("my-model", "my-data", epochs=2, backend="keras")
    assert res["action"] == "posttrain"
    assert res["model"] == "my-model"
    assert res["backend"] == "keras"
    assert res["status"] == "mocked_missing_keras"
    assert res["dataset"] == "my-data"
    assert res["epochs"] == 2


def test_unknown_backend() -> None:
    """Test routing to unknown backend."""
    import pytest

    with pytest.raises(ValueError, match="Unknown backend: missing"):
        train_from_scratch("my-model", "my-data", backend="missing")

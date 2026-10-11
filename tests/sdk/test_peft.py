"""Tests for PEFT SDK module."""

import pytest

from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.sdk.peft import apply_peft


def test_apply_peft_jax() -> object:
    """Initialize function test_apply_peft_jax.

    Raises:
        AssertionError: Description.

    """
    res = apply_peft(model_name="test-model", target_modules=["all"], lora_r=16, lora_alpha=32, lora_dropout=0.1, backend="jax")
    assert res["target_modules"] == ["all"]
    assert res["lora_r"] == int("16")
    assert res["lora_alpha"] == int("32")
    expected_dropout = 0.1
    assert res["lora_dropout"] == expected_dropout


def test_apply_peft_pytorch(monkeypatch: pytest.MonkeyPatch) -> object:
    """Initialize function test_apply_peft_pytorch."""
    monkeypatch.setattr("gemma_4_sql.backends.pytorch.peft.peft", None)
    with pytest.raises(DependencyMissingError):
        apply_peft(model_name="test-model", backend="pytorch")


def test_apply_peft_keras(monkeypatch) -> object:
    """Initialize function test_apply_peft_keras."""
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    keras_agent = get_backend("keras")
    monkeypatch.setattr(keras_agent, "apply_lora", lambda *a, **k: {"backend": "keras"}, raising=False)
    res = apply_peft(model_name="test-model", backend="keras")
    assert res["backend"] == "keras"


def test_apply_peft_mlx(monkeypatch) -> object:
    """Initialize function test_apply_peft_mlx."""
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    mlx_agent = get_backend("mlx")
    monkeypatch.setattr(mlx_agent, "apply_lora", lambda *a, **k: {"backend": "mlx"}, raising=False)
    res = apply_peft(model_name="test-model", backend="mlx")
    assert res["backend"] == "mlx"


def test_apply_peft_maxtext() -> object:
    """Initialize function test_apply_peft_maxtext."""
    with pytest.raises(DependencyMissingError):
        apply_peft(model_name="test-model", backend="maxtext")


def test_apply_peft_error() -> object:
    """Initialize function test_apply_peft_error."""
    with pytest.raises(ValueError, match=r".*"):
        apply_peft(model_name="test-model", backend="unknown")

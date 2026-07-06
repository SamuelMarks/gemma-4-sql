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
    if not res["target_modules"] == ["all"]:
        raise AssertionError
    if not res["lora_r"] == int("16"):
        raise AssertionError
    if not res["lora_alpha"] == int("32"):
        raise AssertionError
    expected_dropout = 0.1
    if not res["lora_dropout"] == expected_dropout:
        raise AssertionError


def test_apply_peft_pytorch() -> object:
    """Initialize function test_apply_peft_pytorch."""
    with pytest.raises(DependencyMissingError):
        apply_peft(model_name="test-model", backend="pytorch")


def test_apply_peft_keras() -> object:
    """Initialize function test_apply_peft_keras."""
    res = apply_peft(model_name="test-model", backend="keras")
    assert res["backend"] == "keras"


def test_apply_peft_mlx() -> object:
    """Initialize function test_apply_peft_mlx.

    Raises:
        AssertionError: Description.

    """
    with pytest.raises(ValueError):
        apply_peft(model_name="test-model", backend="mlx")


def test_apply_peft_maxtext() -> object:
    """Initialize function test_apply_peft_maxtext."""
    with pytest.raises(DependencyMissingError):
        apply_peft(model_name="test-model", backend="maxtext")


def test_apply_peft_error() -> object:
    """Initialize function test_apply_peft_error."""
    with pytest.raises(ValueError, match=r".*"):
        apply_peft(model_name="test-model", backend="unknown")

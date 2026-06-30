"""Tests for PEFT SDK module."""

import pytest

from gemma_4_sql.sdk.peft import apply_peft


def test_apply_peft_jax() -> object:  # type: ignore[return]
    """Initialize function test_apply_peft_jax."""
    res = apply_peft(model_name="test-model", backend="jax")
    if not res["backend"] == "jax":
        raise AssertionError
    if not res["model"] == "test-model":
        raise AssertionError
    if not res["target_modules"] == ["q_proj", "v_proj"]:
        raise AssertionError
    if not res["lora_r"] == int("8"):
        raise AssertionError
    if not res["lora_alpha"] == int("16"):
        raise AssertionError
    expected_dropout = 0.05
    if not res["lora_dropout"] == expected_dropout:
        raise AssertionError
    if "status" not in res:
        raise AssertionError


def test_apply_peft_keras() -> object:  # type: ignore[return]
    """Initialize function test_apply_peft_keras."""
    res = apply_peft(model_name="test-model", backend="keras")
    if not res["backend"] == "keras":
        raise AssertionError


def test_apply_peft_maxtext() -> object:  # type: ignore[return]
    """Initialize function test_apply_peft_maxtext."""
    res = apply_peft(model_name="test-model", backend="maxtext")
    if not res["backend"] == "maxtext":
        raise AssertionError


def test_apply_peft_pytorch() -> object:  # type: ignore[return]
    """Initialize function test_apply_peft_pytorch."""
    res = apply_peft(model_name="test-model", backend="pytorch")
    if not res["backend"] == "pytorch":
        raise AssertionError


def test_apply_peft_unknown_backend() -> object:  # type: ignore[return]
    """Initialize function test_apply_peft_unknown_backend."""
    with pytest.raises(ValueError, match="Unknown backend: unknown"):
        apply_peft(model_name="test-model", backend="unknown")


def test_apply_peft_custom_params() -> object:  # type: ignore[return]
    """Initialize function test_apply_peft_custom_params."""
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

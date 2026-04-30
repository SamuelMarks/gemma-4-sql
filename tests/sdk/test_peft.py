"""Tests for PEFT SDK module."""

import pytest

from gemma_4_sql.sdk.peft import apply_peft


def test_apply_peft_jax():
    res = apply_peft(model_name="test-model", backend="jax")
    assert res["backend"] == "jax"
    assert res["model"] == "test-model"
    assert res["target_modules"] == ["q_proj", "v_proj"]
    assert res["lora_r"] == 8
    assert res["lora_alpha"] == 16
    assert res["lora_dropout"] == 0.05
    assert "status" in res


def test_apply_peft_keras():
    res = apply_peft(model_name="test-model", backend="keras")
    assert res["backend"] == "keras"


def test_apply_peft_maxtext():
    res = apply_peft(model_name="test-model", backend="maxtext")
    assert res["backend"] == "maxtext"


def test_apply_peft_pytorch():
    res = apply_peft(model_name="test-model", backend="pytorch")
    assert res["backend"] == "pytorch"


def test_apply_peft_unknown_backend():
    with pytest.raises(ValueError, match="Unknown backend: unknown"):
        apply_peft(model_name="test-model", backend="unknown")


def test_apply_peft_custom_params():
    res = apply_peft(
        model_name="test-model",
        target_modules=["all"],
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        backend="jax",
    )
    assert res["target_modules"] == ["all"]
    assert res["lora_r"] == 16
    assert res["lora_alpha"] == 32
    assert res["lora_dropout"] == 0.1

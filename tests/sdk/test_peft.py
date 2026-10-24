"""Behavioral tests for the PEFT / LoRA SDK module."""

from __future__ import annotations

from typing import Any

import pytest

from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.sdk.peft import apply_peft
from gemma_4_sql.sdk.registry import get_backend


def test_apply_peft_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify apply_peft injects LoRA parameters on the JAX backend and returns adapter metadata."""
    import gemma_4_sql.backends.jax.peft as jax_peft
    from gemma_4_sql.backends.jax.gemma4.config import ModelConfig

    tiny_cfg = ModelConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=1,
    )
    monkeypatch.setattr(jax_peft.Gemma4Config, "gemma4_e2b", staticmethod(lambda: tiny_cfg))
    res = apply_peft(
        model_name="test-model",
        target_modules=["q_proj", "v_proj"],
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        backend="jax",
    )
    assert res["backend"] == "jax"
    assert res["target_modules"] == ["q_proj", "v_proj"]
    assert res["lora_r"] == 16
    assert res["lora_alpha"] == 32
    assert res["lora_dropout"] == pytest.approx(0.1)
    assert int(str(res.get("injected_modules"))) == 2


def test_apply_peft_pytorch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify apply_peft raises DependencyMissingError when PyTorch PEFT is unavailable."""
    monkeypatch.setattr("gemma_4_sql.backends.pytorch.peft.peft", None)
    with pytest.raises(DependencyMissingError):
        apply_peft(model_name="test-model", backend="pytorch")


def test_apply_peft_keras(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify apply_peft routes default target_modules and hyperparameters to Keras backend."""
    keras_agent = get_backend("keras")
    captured: dict[str, Any] = {}

    def spy_lora(
        model_name: str,
        target_modules: list[str],
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        """Capture forwarded LoRA arguments."""
        captured["model_name"] = model_name
        captured["target_modules"] = target_modules
        captured["lora_r"] = lora_r
        captured["lora_alpha"] = lora_alpha
        captured["lora_dropout"] = lora_dropout
        return {"backend": "keras", "target_modules": target_modules, "lora_r": lora_r}

    monkeypatch.setattr(keras_agent, "apply_lora", spy_lora)
    res = apply_peft(model_name="test-model", backend="keras")
    assert res["backend"] == "keras"
    assert captured["model_name"] == "test-model"
    assert captured["target_modules"] == ["q_proj", "v_proj"]
    assert captured["lora_r"] == 8
    assert captured["lora_alpha"] == 16


def test_apply_peft_mlx(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify apply_peft routes custom target_modules to the MLX backend."""
    mlx_agent = get_backend("mlx")
    captured: dict[str, Any] = {}

    def spy_lora(
        model_name: str,
        target_modules: list[str],
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        """Capture forwarded MLX LoRA arguments."""
        captured["target_modules"] = target_modules
        return {"backend": "mlx", "target_modules": target_modules}

    monkeypatch.setattr(mlx_agent, "apply_lora", spy_lora)
    res = apply_peft(model_name="test-model", target_modules=["k_proj"], backend="mlx")
    assert res["backend"] == "mlx"
    assert captured["target_modules"] == ["k_proj"]


def test_apply_peft_maxtext() -> None:
    """Verify apply_peft raises DependencyMissingError when MaxText is unavailable."""
    with pytest.raises(DependencyMissingError):
        apply_peft(model_name="test-model", backend="maxtext")


def test_apply_peft_error() -> None:
    """Verify apply_peft raises ValueError on an unknown backend."""
    with pytest.raises(ValueError, match="Unknown backend: unknown"):
        apply_peft(model_name="test-model", backend="unknown")

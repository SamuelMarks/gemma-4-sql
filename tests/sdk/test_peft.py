"""Behavioral tests for the PEFT / LoRA SDK module."""

from __future__ import annotations

from typing import Any

import pytest

from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.sdk.peft import apply_peft


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
    """Verify apply_peft executes Keras apply_lora and injects LoRA into target modules."""
    import gemma_4_sql.backends.keras.peft as kpeft

    class FakeBackbone:
        """Keras backbone stub supporting enable_lora."""

        def __init__(self) -> None:
            """Initialize backbone stub."""
            self.enabled_rank = 0
            self.layers: list[Any] = []

        def enable_lora(self, rank: int) -> None:
            """Record enabled LoRA rank."""
            self.enabled_rank = rank

    class FakeKerasModel:
        """Keras causal LM stub with backbone."""

        def __init__(self) -> None:
            """Initialize model stub."""
            self.backbone = FakeBackbone()

    backbone_holder = FakeKerasModel()

    class FakeGemmaCausalLM:
        """Stub for GemmaCausalLM.from_preset."""

        @staticmethod
        def from_preset(_name: str) -> FakeKerasModel:
            """Return the FakeKerasModel instance."""
            return backbone_holder

    monkeypatch.setattr(kpeft, "keras", object())
    monkeypatch.setattr(kpeft, "optional_attr", lambda *_a: FakeGemmaCausalLM)
    res = apply_peft(model_name="test-model", backend="keras", lora_r=12)
    assert res["backend"] == "keras"
    assert res["target_modules"] == ["q_proj", "v_proj"]
    assert res["lora_r"] == 12
    assert res["injected_modules"] == 2
    assert backbone_holder.backbone.enabled_rank == 12


def test_apply_peft_mlx(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify apply_peft routes custom target_modules through MLX apply_lora."""
    import gemma_4_sql.backends.mlx.peft as mpeft

    injected_targets: list[str] = []

    def _fake_inject_lora(
        model: Any,
        target_modules: list[str],
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
    ) -> tuple[Any, int]:
        """Record target modules and return injected count."""
        _ = (lora_r, lora_alpha, lora_dropout)
        injected_targets.extend(target_modules)
        return (model, len(target_modules))

    monkeypatch.setattr(mpeft, "nn", object())
    monkeypatch.setattr(mpeft, "mx", object())
    monkeypatch.setattr(mpeft, "load", lambda _m: (object(), None))
    monkeypatch.setattr(mpeft, "inject_lora", _fake_inject_lora)

    res = apply_peft(model_name="test-model", target_modules=["k_proj"], backend="mlx")
    assert res["backend"] == "mlx"
    assert res["target_modules"] == ["k_proj"]
    assert res["injected_modules"] == 1
    assert injected_targets == ["k_proj"]


def test_apply_peft_maxtext() -> None:
    """Verify apply_peft raises DependencyMissingError when MaxText is unavailable."""
    with pytest.raises(DependencyMissingError):
        apply_peft(model_name="test-model", backend="maxtext")


def test_apply_peft_error() -> None:
    """Verify apply_peft raises ValueError on an unknown backend."""
    with pytest.raises(ValueError, match="Unknown backend: unknown"):
        apply_peft(model_name="test-model", backend="unknown")

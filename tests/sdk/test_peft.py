"""Behavioral tests for the PEFT / LoRA SDK module."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from gemma_4_sql.exceptions import ConfigurationError, DependencyMissingError
from gemma_4_sql.sdk.peft import apply_peft
from gemma_4_sql.type_hints import KerasLayerProtocol


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
    """Verify apply_peft injects real NativeLoRALinear layers on pytorch_native and raises DependencyMissingError when HF peft is unavailable."""
    import torch

    from gemma_4_sql.backends.pytorch.gemma4.config import Gemma4Config
    from gemma_4_sql.backends.pytorch.gemma4.modeling import Gemma4ForCausalLM
    from gemma_4_sql.backends.pytorch.peft import inject_native_pytorch_lora

    tiny_cfg = Gemma4Config.gemma4_tiny()
    tiny_model = Gemma4ForCausalLM(tiny_cfg)
    base_param_count = sum(p.numel() for p in tiny_model.parameters())

    res_native = apply_peft(
        model_name="gemma-4-tiny",
        target_modules=["q_proj", "v_proj"],
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        backend="pytorch_native",
    )
    assert res_native["backend"] == "pytorch_native"
    assert res_native["status"] == "completed"
    assert int(res_native.get("injected_modules", 0)) >= 2

    injected_count = inject_native_pytorch_lora(tiny_model, ["q_proj", "v_proj"], lora_r=4, lora_alpha=8, lora_dropout=0.0)
    assert injected_count >= 2
    lora_param_count = sum(p.numel() for p in tiny_model.parameters())
    assert lora_param_count > base_param_count
    assert any(type(m).__name__ == "NativeLoRALinear" for _, m in tiny_model.named_modules())
    out = tiny_model(torch.tensor([[1, 2, 3]], dtype=torch.long))
    logits = out[0] if isinstance(out, tuple) else out
    assert logits.shape == (1, 3, tiny_cfg.vocab_size)

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
            self.layers: list[KerasLayerProtocol] = []

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

    class DummyKerasSentinel:
        """Non-None sentinel for keras module check."""

    def _resolve_fake_causal_lm(*_args: str | type) -> type[FakeGemmaCausalLM]:
        """Return FakeGemmaCausalLM class."""
        return FakeGemmaCausalLM

    monkeypatch.setattr(kpeft, "keras", DummyKerasSentinel())
    monkeypatch.setattr(kpeft, "optional_attr", _resolve_fake_causal_lm)
    res = apply_peft(model_name="test-model", backend="keras", lora_r=12)
    assert res["backend"] == "keras"
    assert res["target_modules"] == ["q_proj", "v_proj"]
    assert res["lora_r"] == 12
    assert res["injected_modules"] == 2
    assert backbone_holder.backbone.enabled_rank == 12


def test_apply_peft_mlx(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify apply_peft routes custom target_modules through MLX apply_lora and injects real MLXLoRALinear modules."""
    import gemma_4_sql.backends.mlx.peft as mpeft

    class DummyMlxModel:
        """Minimal MLX model stub."""

    injected_targets: list[str] = []

    def _fake_inject_lora(
        model: DummyMlxModel,
        target_modules: list[str],
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
    ) -> tuple[DummyMlxModel, int]:
        """Record target modules and return injected count."""
        _ = (lora_r, lora_alpha, lora_dropout)
        injected_targets.extend(target_modules)
        return (model, len(target_modules))

    def _fake_load(_name: str) -> tuple[DummyMlxModel, None]:
        """Return DummyMlxModel and None tokenizer."""
        return (DummyMlxModel(), None)

    orig_inject_lora = mpeft.inject_lora
    monkeypatch.setattr(mpeft, "load", _fake_load)
    monkeypatch.setattr(mpeft, "inject_lora", _fake_inject_lora)

    res = apply_peft(model_name="test-model", target_modules=["k_proj"], backend="mlx")
    assert res["backend"] == "mlx"
    assert res["target_modules"] == ["k_proj"]
    assert res["injected_modules"] == 1
    assert injected_targets == ["k_proj"]

    import mlx.core as mx
    from mlx import nn

    from gemma_4_sql.backends.mlx.peft import MLXLoRALinear

    monkeypatch.setattr(mpeft, "inject_lora", orig_inject_lora)

    class _TinyMlxAttn(nn.Module):
        """Tiny MLX attention block with q_proj and v_proj Linear layers."""

        def __init__(self) -> None:
            """Initialize linear projections."""
            super().__init__()
            self.q_proj = nn.Linear(8, 8)
            self.v_proj = nn.Linear(8, 8)

        def __call__(self, x: mx.array) -> mx.array:
            """Compute sum of q_proj and v_proj."""
            q_fn: Callable[[mx.array], mx.array] = self.q_proj
            v_fn: Callable[[mx.array], mx.array] = self.v_proj
            return q_fn(x) + v_fn(x)

    class _TinyMlxModel(nn.Module):
        """Tiny MLX model wrapping _TinyMlxAttn."""

        def __init__(self) -> None:
            """Initialize nested attention block."""
            super().__init__()
            self.attn = _TinyMlxAttn()

        def __call__(self, x: mx.array) -> mx.array:
            """Forward through nested attention block."""
            return self.attn(x)

    tiny_mlx = _TinyMlxModel()

    def _load_tiny_mlx(_name: str) -> tuple[_TinyMlxModel, None]:
        """Return `(tiny_mlx, None)`."""
        return (tiny_mlx, None)

    monkeypatch.setattr(mpeft, "load", _load_tiny_mlx)
    res_real = apply_peft(
        model_name="gemma-4-tiny",
        target_modules=["q_proj", "v_proj"],
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        backend="mlx",
    )
    assert res_real["backend"] == "mlx"
    assert res_real["injected_modules"] == 2
    assert isinstance(tiny_mlx.attn.q_proj, MLXLoRALinear)
    assert isinstance(tiny_mlx.attn.v_proj, MLXLoRALinear)
    sample_out = tiny_mlx(mx.ones((1, 3, 8)))
    assert tuple(sample_out.shape) == (1, 3, 8)


def test_apply_peft_maxtext() -> None:
    """Verify apply_peft raises DependencyMissingError when MaxText is unavailable."""
    with pytest.raises(DependencyMissingError):
        apply_peft(model_name="test-model", backend="maxtext")


def test_apply_peft_error() -> None:
    """Verify apply_peft raises ConfigurationError on an unknown backend."""
    with pytest.raises(ConfigurationError, match="Unknown backend: unknown"):
        apply_peft(model_name="test-model", backend="unknown")

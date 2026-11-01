"""Behavioral tests for the PEFT / LoRA SDK module."""

from __future__ import annotations

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
    from typing import Self

    import gemma_4_sql.backends.mlx.peft as mpeft
    from gemma_4_sql.backends.mlx.peft import MLXLoRALinear

    class _TinyMlxArray:
        """Minimal 2D/3D tensor double satisfying `_MlxTensorProtocol` for cross-platform `MLXLoRALinear` tests."""

        def __init__(self, shape: tuple[int, ...]) -> None:
            """Initialize with target shape."""
            self._shape = shape

        @property
        def shape(self) -> tuple[int, ...]:
            """Provide tensor shape."""
            return self._shape

        @property
        def T(self) -> Self:
            """Provide transposed 2D shape."""
            return type(self)((self._shape[1], self._shape[0]))

        def flatten(self) -> Self:
            """Return 1D flattened shape."""
            total = 1
            for d in self._shape:
                total *= d
            return type(self)((total,))

        def tolist(self) -> list[float]:
            """Return zero-filled float list."""
            return [0.0]

        def __add__(self, _other: Self | float, /) -> Self:
            """Return array of same shape under addition."""
            return type(self)(self._shape)

        def __radd__(self, _other: Self | float, /) -> Self:
            """Return array of same shape under right-addition."""
            return type(self)(self._shape)

        def __mul__(self, _other: Self | float, /) -> Self:
            """Return array of same shape under multiplication."""
            return type(self)(self._shape)

        def __rmul__(self, _other: Self | float, /) -> Self:
            """Return array of same shape under right-multiplication."""
            return type(self)(self._shape)

        def __matmul__(self, other: Self, /) -> Self:
            """Return matrix product shape `(..., other.shape[-1])`."""
            return type(self)((*self._shape[:-1], other.shape[-1]))

    class _TinyMlxRandom:
        """Minimal `mx.random` namespace providing `uniform`."""

        def uniform(self, low: float, high: float, shape: tuple[int, ...]) -> _TinyMlxArray:
            """Return a `_TinyMlxArray` with the requested shape."""
            _ = (low, high)
            return _TinyMlxArray(shape)

    class _TinyMlxCore:
        """Minimal `mlx.core` double providing `zeros` and `random.uniform`."""

        def __init__(self) -> None:
            """Initialize random namespace."""
            self.random = _TinyMlxRandom()

        def zeros(self, shape: tuple[int, ...]) -> _TinyMlxArray:
            """Return a zero-initialized `_TinyMlxArray`."""
            return _TinyMlxArray(shape)

    class _TinyLinear:
        """Minimal `mlx.nn.Linear` double exposing `weight` and `bias`."""

        def __init__(self, in_dims: int, out_dims: int) -> None:
            """Initialize weight matrix and `None` bias."""
            self.weight = _TinyMlxArray((out_dims, in_dims))
            self.bias: _TinyMlxArray | None = None

        def __call__(self, x: _TinyMlxArray) -> _TinyMlxArray:
            """Project `x` by `weight.T`."""
            return x @ self.weight.T

    class _TinyIdentity:
        """Minimal `mlx.nn.Identity` pass-through callable."""

        def __call__(self, x: _TinyMlxArray) -> _TinyMlxArray:
            """Return `x` unchanged."""
            return x

    class _TinyMlxNN:
        """Minimal `mlx.nn` namespace exposing `Linear` and `Identity`."""

        Linear = _TinyLinear
        Identity = _TinyIdentity

    monkeypatch.setattr(mpeft, "mx", _TinyMlxCore())
    monkeypatch.setattr(mpeft, "nn", _TinyMlxNN())

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

    monkeypatch.setattr(mpeft, "inject_lora", orig_inject_lora)

    class _TinyMlxAttn(dict[str, _TinyLinear | MLXLoRALinear]):
        """Tiny MLX attention block (`dict` subclass matching `mlx.nn.Module`) with `q_proj` and `v_proj`."""

        def __init__(self) -> None:
            """Initialize linear projections."""
            super().__init__()
            self["q_proj"] = _TinyLinear(8, 8)
            self["v_proj"] = _TinyLinear(8, 8)

        def __call__(self, x: _TinyMlxArray) -> _TinyMlxArray:
            """Compute sum of `q_proj` and `v_proj`."""
            return self["q_proj"](x) + self["v_proj"](x)

    class _TinyMlxModel(dict[str, _TinyMlxAttn]):
        """Tiny MLX model wrapping `_TinyMlxAttn` and exposing `named_modules()`."""

        def __init__(self) -> None:
            """Initialize nested attention block."""
            super().__init__()
            self["attn"] = _TinyMlxAttn()

        def named_modules(self) -> list[tuple[str, _TinyLinear | MLXLoRALinear]]:
            """Yield dotted module paths for `q_proj` and `v_proj`."""
            attn = self["attn"]
            return [("attn.q_proj", attn["q_proj"]), ("attn.v_proj", attn["v_proj"])]

        def __call__(self, x: _TinyMlxArray) -> _TinyMlxArray:
            """Forward through nested attention block."""
            return self["attn"](x)

    tiny_mlx = _TinyMlxModel()
    pre_linear = _TinyLinear(8, 8)
    assert pre_linear(_TinyMlxArray((1, 3, 8))).shape == (1, 3, 8)
    assert pre_linear.weight.flatten().tolist() == [0.0]
    assert (_TinyMlxArray((2, 2)) + 1.0).shape == (2, 2)
    assert (1.0 + _TinyMlxArray((2, 2))).shape == (2, 2)
    assert (_TinyMlxArray((2, 2)) * 2.0).shape == (2, 2)

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
    q_lora = tiny_mlx["attn"]["q_proj"]
    v_lora = tiny_mlx["attn"]["v_proj"]
    assert isinstance(q_lora, MLXLoRALinear)
    assert isinstance(v_lora, MLXLoRALinear)
    assert set(q_lora.parameters()) == {"weight", "lora_a", "lora_b"}
    assert set(q_lora.trainable_parameters()) == {"lora_a", "lora_b"}
    biased_lora = MLXLoRALinear(in_features=8, out_features=8, r=2, lora_alpha=4.0, lora_dropout=0.0, bias=True)
    assert set(biased_lora.parameters()) == {"weight", "bias", "lora_a", "lora_b"}
    sample_out = tiny_mlx(_TinyMlxArray((1, 3, 8)))
    assert sample_out.shape == (1, 3, 8)


def test_apply_peft_maxtext() -> None:
    """Verify apply_peft raises DependencyMissingError when MaxText is unavailable."""
    with pytest.raises(DependencyMissingError):
        apply_peft(model_name="test-model", backend="maxtext")


def test_apply_peft_error() -> None:
    """Verify apply_peft raises ConfigurationError on an unknown backend."""
    with pytest.raises(ConfigurationError, match="Unknown backend: unknown"):
        apply_peft(model_name="test-model", backend="unknown")

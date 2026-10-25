"""Tests for JAX PEFT / LoRA implementation."""

from __future__ import annotations

from tests.conftest import requires_jax

pytestmark = requires_jax

from typing import Any

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

import gemma_4_sql.backends.jax.peft as pt
from gemma_4_sql.backends.jax.gemma4 import Gemma4Config, Gemma4ForCausalLM
from gemma_4_sql.backends.jax.peft import (
    apply_lora,
    count_parameters,
    create_lora_optimizer,
    inject_lora,
    inject_lora_to_layer,
)
from gemma_4_sql.exceptions import DependencyMissingError, TrainingError


def test_apply_lora_jax_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test apply_lora raises DependencyMissingError when optax is missing."""
    monkeypatch.setattr(pt, "optax", None)
    with pytest.raises(DependencyMissingError, match=r"JAX PEFT dependencies are missing\."):
        pt.apply_lora("test-model", ["q_proj"], 8, 16, 0.05)


def test_apply_lora_jax_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test apply_lora captures exceptions during execution and respects raise_on_error."""
    monkeypatch.setattr(Gemma4Config, "gemma4_e2b", Gemma4Config.gemma4_tiny)
    res_default = pt.apply_lora("test-model", ["q_proj"], 2, 4, 0.0)
    assert res_default["status"] == "completed"
    assert res_default.get("injected_modules") == 1

    def mock_inject(*_args: Any, **_kwargs: Any) -> tuple[Any, int]:
        """Raise intentional ValueError to simulate injection error."""
        raise ValueError("split error")

    monkeypatch.setattr(pt, "inject_lora", mock_inject)
    res = pt.apply_lora("test-model", ["q_proj"], 8, 16, 0.05)
    assert "failed" in str(res["status"])
    with pytest.raises(TrainingError, match="split error"):
        pt.apply_lora("test-model", ["q_proj"], 8, 16, 0.05, raise_on_error=True)


def test_nnx_lora_linear_init_and_properties() -> None:
    """Test initialization, attributes, and property accessors of NNXLoRALinear."""
    rngs = nnx.Rngs(params=0, dropout=1)
    layer = pt.NNXLoRALinear(
        in_features=8,
        out_features=16,
        r=4,
        lora_alpha=8.0,
        lora_dropout=0.1,
        use_bias=True,
        rngs=rngs,
    )
    assert layer.in_features == 8
    assert layer.out_features == 16
    assert layer.r == 4
    assert layer.lora_alpha == 8.0
    assert layer.scale == 2.0
    assert layer.kernel.shape == (8, 16)
    assert layer.bias is not None
    assert layer.bias.shape == (16,)
    assert layer.lora_a.shape == (8, 4)
    assert layer.lora_b.shape == (4, 16)
    assert isinstance(layer.lora_a, pt.LoRAParam)
    assert isinstance(layer.lora_b, pt.LoRAParam)

    # Test properties W, A, B
    assert jnp.array_equal(layer.W, layer.kernel[...])
    assert jnp.array_equal(layer.A, layer.lora_a[...])
    assert jnp.array_equal(layer.B, layer.lora_b[...])


def test_nnx_lora_linear_zero_dropout_and_default_rngs() -> None:
    """Test NNXLoRALinear with 0 dropout and default rngs initialization."""
    layer = pt.NNXLoRALinear(in_features=4, out_features=4, r=2, lora_dropout=0.0, use_bias=False)
    assert layer.bias is None
    assert layer.dropout is None


def test_nnx_lora_linear_invalid_rank() -> None:
    """Test NNXLoRALinear raises ValueError for rank <= 0."""
    with pytest.raises(ValueError, match="LoRA rank r must be positive"):
        pt.NNXLoRALinear(in_features=4, out_features=4, r=0)

    with pytest.raises(ValueError, match="LoRA rank r must be positive"):
        pt.NNXLoRALinear(in_features=4, out_features=4, r=-2)


def test_nnx_lora_linear_mode_switching() -> None:
    """Test train() and eval() mode switching on NNXLoRALinear."""
    rngs = nnx.Rngs(params=0, dropout=1)
    layer = pt.NNXLoRALinear(in_features=4, out_features=4, r=2, lora_dropout=0.5, rngs=rngs)
    assert layer.dropout is not None

    layer.eval()
    x = jnp.ones((2, 4))
    out_eval1 = layer(x)
    out_eval2 = layer(x)
    assert jnp.array_equal(out_eval1, out_eval2)

    layer.train(True)
    layer.train(False)

    # Calling train/eval on layer with no dropout is a safe no-op
    layer_no_drop = pt.NNXLoRALinear(in_features=4, out_features=4, r=2, lora_dropout=0.0)
    layer_no_drop.train(True)
    layer_no_drop.eval()


def test_nnx_lora_linear_from_linear() -> None:
    """Test constructing NNXLoRALinear from an existing nnx.Linear layer."""
    rngs = nnx.Rngs(0)
    lin_bias = nnx.Linear(8, 12, use_bias=True, rngs=rngs)
    lora_bias = pt.NNXLoRALinear.from_linear(lin_bias, r=4, lora_alpha=16.0, lora_dropout=0.0, rngs=rngs)
    assert lora_bias.in_features == 8
    assert lora_bias.out_features == 12
    assert lora_bias.kernel is lin_bias.kernel
    assert lora_bias.bias is lin_bias.bias

    lin_no_bias = nnx.Linear(6, 10, use_bias=False, rngs=rngs)
    lora_no_bias = pt.NNXLoRALinear.from_linear(lin_no_bias, r=2, rngs=rngs)
    assert lora_no_bias.bias is None
    assert lora_no_bias.kernel is lin_no_bias.kernel


def test_nnx_lora_linear_forward_pass() -> None:
    """Test mathematical correctness of NNXLoRALinear forward pass."""
    rngs = nnx.Rngs(params=0, dropout=1)
    layer = pt.NNXLoRALinear(in_features=4, out_features=4, r=2, lora_alpha=4.0, lora_dropout=0.0, use_bias=True, rngs=rngs)
    layer.kernel[...] = jnp.ones((4, 4))
    assert layer.bias is not None
    layer.bias[...] = jnp.full((4,), 2.0)
    layer.lora_a[...] = jnp.ones((4, 2))
    layer.lora_b[...] = jnp.zeros((2, 4))

    x = jnp.ones((2, 4))
    # When lora_b is zero: y = x @ W + bias = 4 * 1.0 + 2.0 = 6.0
    y = layer(x)
    assert jnp.allclose(y, jnp.full((2, 4), 6.0))

    # When lora_b is ones:
    # (x @ A) = 4 in shape (2, 2)
    # (x @ A) @ B = 8 in shape (2, 4)
    # lora_term = scale * 8 = (4 / 2) * 8 = 16.0
    # y = 6.0 + 16.0 = 22.0
    layer.lora_b[...] = jnp.ones((2, 4))
    y_with_lora = layer(x, deterministic=True)
    assert jnp.allclose(y_with_lora, jnp.full((2, 4), 22.0))

    layer.bias = type("Param", (), {"value": None})()
    y_none_bias = layer(x, deterministic=True)
    assert y_none_bias is not None


def test_inject_lora_to_decoder_layer() -> None:
    """Test injecting LoRA into all 7 projection layers of Gemma4DecoderLayer."""
    from gemma_4_sql.backends.jax.gemma4.config import AttentionType, ModelConfig

    config = ModelConfig(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        num_hidden_layers=1,
        num_experts=1,
        share_kv_projections=False,
    )
    rngs = nnx.Rngs(0)
    layer = pt.Gemma4DecoderLayer(config, AttentionType.LOCAL_SLIDING, rngs=rngs)

    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    count = inject_lora_to_layer(layer, targets, lora_r=2, lora_alpha=4.0, rngs=rngs)
    assert count == 7

    assert isinstance(layer.self_attention.q_proj, pt.NNXLoRALinear)
    assert isinstance(layer.self_attention.k_proj, pt.NNXLoRALinear)
    assert isinstance(layer.self_attention.v_proj, pt.NNXLoRALinear)
    assert isinstance(layer.self_attention.o_proj, pt.NNXLoRALinear)
    assert isinstance(layer.mlp.gate_proj, pt.NNXLoRALinear)
    assert isinstance(layer.mlp.up_proj, pt.NNXLoRALinear)
    assert isinstance(layer.mlp.down_proj, pt.NNXLoRALinear)

    # Test when self_attention has share_kv_projections (v_proj is None)
    config_shared = ModelConfig(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        num_hidden_layers=1,
        num_experts=1,
        share_kv_projections=True,
    )
    layer_shared = pt.Gemma4DecoderLayer(config_shared, AttentionType.GLOBAL, rngs=rngs)
    count_shared = inject_lora_to_layer(layer_shared, targets, lora_r=2, rngs=rngs)
    # v_proj is None so only 6 are injected
    assert count_shared == 6

    # Test layer with non-Linear projection attribute
    layer_with_fake: Any = pt.Gemma4DecoderLayer(config, AttentionType.LOCAL_SLIDING, rngs=rngs)
    layer_with_fake.mlp.gate_proj = "not_a_linear"
    assert inject_lora_to_layer(layer_with_fake, ["gate_proj"]) == 0

    # Test layer with no self_attention or mlp
    dummy = object()
    assert inject_lora_to_layer(dummy, targets) == 0


def test_inject_lora_full_model_and_counting() -> None:
    """Test inject_lora on Gemma4ForCausalLM and parameter counting."""
    config = Gemma4Config.gemma4_e2b()
    config.num_hidden_layers = 2
    rngs = nnx.Rngs(0)
    model: Any = Gemma4ForCausalLM(config, rngs=rngs)

    targets = ["q_proj", "v_proj"]
    adapted_model, count = inject_lora(model, targets, lora_r=4, rngs=rngs)
    # 2 layers * 2 targets = 4 injected LoRA modules
    assert count == 4

    total_params, lora_params = count_parameters(adapted_model)
    assert total_params > lora_params
    assert lora_params > 0

    # Test inject_lora with empty target list
    m, c = inject_lora(model, [])
    assert c == 0
    assert m is model

    # Test inject_lora on object without __dict__
    prim_obj: Any = 123
    _m2, c2 = inject_lora(prim_obj, ["q_proj"])
    assert c2 == 0


def test_inject_lora_moe_shared_experts() -> None:
    """Test inject_lora_to_layer on MoE decoder layer with shared_experts."""
    from gemma_4_sql.backends.jax.gemma4.config import AttentionType, ModelConfig

    config = ModelConfig(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        num_hidden_layers=1,
        num_experts=4,
    )
    rngs = nnx.Rngs(0)
    layer = pt.Gemma4DecoderLayer(config, AttentionType.LOCAL_SLIDING, rngs=rngs)
    count = inject_lora_to_layer(layer, ["gate_proj", "up_proj", "down_proj"], lora_r=2, rngs=rngs)
    assert count == 3
    assert isinstance(layer.mlp.shared_experts.gate_proj, pt.NNXLoRALinear)


def test_inject_lora_model_layers_attribute() -> None:
    """Test inject_lora on model with direct layers list attribute."""
    from gemma_4_sql.backends.jax.gemma4.config import AttentionType, ModelConfig

    config = ModelConfig(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        num_hidden_layers=1,
        num_experts=1,
    )
    rngs = nnx.Rngs(0)
    layer = pt.Gemma4DecoderLayer(config, AttentionType.LOCAL_SLIDING, rngs=rngs)

    class DirectLayersModel:
        """Model with direct layers attribute."""

        def __init__(self, l: Any) -> None:
            """Initialize with layer list."""
            self.layers = [l]

    model: Any = DirectLayersModel(layer)
    adapted, count = inject_lora(model, ["q_proj"], lora_r=2, rngs=rngs)
    assert count == 1
    assert isinstance(adapted.layers[0].self_attention.q_proj, pt.NNXLoRALinear)


def test_inject_lora_generic_recursive_traversal() -> None:
    """Test inject_lora fallback recursive traversal on arbitrary module hierarchies."""

    class CustomBlock(nnx.Module):
        """Block containing custom sub-modules and linear layers."""

        def __init__(self, rngs: nnx.Rngs) -> None:
            """Initialize custom linear layers."""
            self.q_proj = nnx.Linear(8, 8, rngs=rngs)
            self.other = nnx.Linear(8, 8, rngs=rngs)

    class PlainContainer:
        """Plain Python class containing raw list and dict of modules."""

        child_module: Any

        def __init__(self, rngs: nnx.Rngs) -> None:
            """Initialize lists and dicts of blocks."""
            self.child_module = CustomBlock(rngs)
            self.child_module.cycle = self
            self.raw_list = [CustomBlock(rngs), 123]
            self.raw_dict = {"item": CustomBlock(rngs), "other": 456}

    rngs = nnx.Rngs(0)
    container: Any = PlainContainer(rngs)
    adapted, count = inject_lora(container, ["q_proj"], lora_r=2, rngs=rngs)
    assert count == 3
    assert isinstance(adapted.child_module.q_proj, pt.NNXLoRALinear)
    assert isinstance(adapted.raw_list[0].q_proj, pt.NNXLoRALinear)
    assert isinstance(adapted.raw_dict["item"].q_proj, pt.NNXLoRALinear)


def test_create_lora_optimizer_default_tx() -> None:
    """Test create_lora_optimizer with default Optax Adam transformation."""
    rngs = nnx.Rngs(0)
    layer = pt.NNXLoRALinear(4, 4, r=2, rngs=rngs)
    opt = create_lora_optimizer(layer)
    assert opt is not None
    assert "lora_a" in str(opt.opt_state)


def test_apply_lora_model_construction() -> None:
    """Test apply_lora constructing model from config and default config."""
    rngs = nnx.Rngs(0)
    config = Gemma4Config.gemma4_e2b()
    config.num_hidden_layers = 1

    # Pass explicit config
    res = apply_lora("gemma-test", ["q_proj"], config=config, rngs=rngs)
    assert res["status"] == "completed"

    # Pass no config and no model
    orig_fn = Gemma4Config.gemma4_e2b
    try:

        def small_config() -> Any:
            """Return small config for fast testing."""
            cfg = orig_fn()
            cfg.num_hidden_layers = 1
            return cfg

        setattr(Gemma4Config, "gemma4_e2b", small_config)
        res_default = apply_lora("gemma-default", ["q_proj"], rngs=rngs)
        assert res_default["status"] == "completed"
    finally:
        setattr(Gemma4Config, "gemma4_e2b", orig_fn)


def test_lora_optimizer_gradient_flow_and_freeze_state() -> None:
    """Test optimizer updates only LoRA adapters while base weights remain frozen."""

    class SimpleModel(nnx.Module):
        """Model with base weights and LoRA adapters for gradient testing."""

        def __init__(self, rngs: nnx.Rngs) -> None:
            """Initialize model."""
            self.proj = pt.NNXLoRALinear(4, 4, r=2, lora_alpha=4.0, lora_dropout=0.0, use_bias=False, rngs=rngs)
            self.proj.kernel[...] = jnp.ones((4, 4))
            self.proj.lora_a[...] = jnp.ones((4, 2))
            self.proj.lora_b[...] = jnp.ones((2, 4))

        def __call__(self, x: jax.Array) -> jax.Array:
            """Forward pass."""
            return self.proj(x)

    rngs = nnx.Rngs(0)
    model: Any = SimpleModel(rngs)
    optimizer = create_lora_optimizer(model, optax.adam(1e-1))

    # Ensure optimizer state tracks LoRAParam and NOT base kernel
    assert "lora_a" in str(optimizer.opt_state)
    assert "lora_b" in str(optimizer.opt_state)

    def loss_fn(m: SimpleModel, x: jax.Array, target: jax.Array) -> jax.Array:
        """Compute MSE loss."""
        pred = m(x)
        return jnp.sum((pred - target) ** 2)

    nnx_any: Any = nnx
    v_ag_fn: Any = nnx_any.value_and_grad
    grad_fn = v_ag_fn(loss_fn, argnums=nnx_any.DiffState(0, pt.LoRAParam))

    x = jnp.ones((1, 4))
    target = jnp.zeros((1, 4))

    from gemma_4_sql.backends.common_train import apply_nnx_optimizer_update

    _loss_before, grads = grad_fn(model, x, target)
    apply_nnx_optimizer_update(optimizer, model, grads)

    # Base kernel must be completely untouched (still all ones)
    assert jnp.all(model.proj.kernel[...] == 1.0)
    # LoRA adapters must have received gradients and updated
    assert not jnp.all(model.proj.lora_a[...] == 1.0)
    assert not jnp.all(model.proj.lora_b[...] == 1.0)


def test_apply_lora_end_to_end() -> None:
    """Test apply_lora end-to-end with real Gemma4 model configuration."""
    config = Gemma4Config.gemma4_e2b()
    config.num_hidden_layers = 1
    rngs = nnx.Rngs(0)
    model = Gemma4ForCausalLM(config, rngs=rngs)

    res = apply_lora(
        model_name="gemma-4-e2b",
        target_modules=["q_proj", "v_proj"],
        lora_r=4,
        lora_alpha=16,
        lora_dropout=0.0,
        model=model,
        rngs=rngs,
    )
    assert res["status"] == "completed"
    assert res["backend"] == "jax"
    assert res.get("injected_modules") == 2
    assert isinstance(model.model.layers[0].self_attention.q_proj, pt.NNXLoRALinear)
    assert isinstance(model.model.layers[0].self_attention.v_proj, pt.NNXLoRALinear)


def test_missing_dependencies_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test functions raise DependencyMissingError when JAX/Flax dependencies are absent."""
    monkeypatch.setattr(pt, "nnx", None)
    dummy_obj: Any = object()

    with pytest.raises(DependencyMissingError, match="JAX PEFT dependencies are missing"):
        pt.NNXLoRALinear(4, 4)

    with pytest.raises(DependencyMissingError, match="JAX PEFT dependencies are missing"):
        pt.inject_lora(dummy_obj, ["q_proj"])

    with pytest.raises(DependencyMissingError, match="JAX PEFT dependencies are missing"):
        pt.create_lora_optimizer(dummy_obj)

    with pytest.raises(DependencyMissingError, match="JAX PEFT dependencies are missing"):
        pt.count_parameters(dummy_obj)

    monkeypatch.setattr(pt, "nnx", nnx)
    monkeypatch.setattr(pt, "optax", None)
    with pytest.raises(DependencyMissingError, match="JAX PEFT dependencies are missing"):
        pt.create_lora_optimizer(dummy_obj)

    monkeypatch.setattr(pt, "optax", optax)
    monkeypatch.setattr(pt, "jax", None)
    with pytest.raises(DependencyMissingError, match="JAX PEFT dependencies are missing"):
        pt.count_parameters(dummy_obj)


def test_lora_linear_no_dropout() -> None:
    """Test NNXLoRALinear forward pass when lora_dropout is 0.0."""
    rngs = nnx.Rngs(params=0, dropout=1)
    layer = pt.NNXLoRALinear(
        in_features=4,
        out_features=4,
        r=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
        use_bias=False,
        rngs=rngs,
    )
    assert layer.dropout is None
    x = jnp.ones((1, 4))
    out = layer(x)
    assert out.shape == (1, 4)


def test_nnx_param_properties_and_indexing() -> None:
    """Test NNXParam and LoRAParam shape, dtype, __getitem__, and __setitem__ behaviors."""
    p = pt.NNXParam(jnp.zeros((2, 3), dtype=jnp.float32))
    assert p.shape == (2, 3)
    assert p.dtype == jnp.float32
    assert p[0].shape == (3,)
    p[...] = jnp.ones((2, 3), dtype=jnp.float32)
    assert float(p.value[0, 0]) == 1.0
    p[0] = jnp.zeros((3,), dtype=jnp.float32)
    assert float(p.value[0, 0]) == 0.0

    # Test when value is a plain Python list (without .at)
    p_list: Any = pt.NNXParam([1, 2, 3])
    p_list[0] = [9, 9, 9]
    assert p_list.value == [9, 9, 9]

    # Test when value is None
    p_empty: Any = pt.NNXParam(None)
    assert p_empty.shape == ()
    assert p_empty.dtype is None
    assert p_empty[0] is None
    p_empty[...] = 5
    assert p_empty.value == 5

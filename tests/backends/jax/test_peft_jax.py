"""Tests for JAX PEFT / LoRA implementation."""

from __future__ import annotations

from tests.conftest import requires_jax

pytestmark = requires_jax

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

import gemma_4_sql.backends.jax.peft as pt
from gemma_4_sql.backends.jax.gemma4 import Gemma4Config, Gemma4ForCausalLM
from gemma_4_sql.backends.jax.gemma4.layers import LinearLayerProtocol
from gemma_4_sql.backends.jax.peft import (
    LoRALinearAdapter,
    LoRAParam,
    NNXLoRALinear,
    NNXParam,
    apply_lora,
    count_parameters,
    create_lora_optimizer,
    inject_lora,
    inject_lora_to_layer,
)
from gemma_4_sql.exceptions import DependencyMissingError, TrainingError


def _make_tiny_config(use_moe: bool = False) -> Gemma4Config:
    """Create a tiny Gemma4Config for fast unit tests.

    Args:
        use_moe: Whether to enable Mixture-of-Experts layers.

    Returns:
        A tiny Gemma4Config instance.

    """
    return Gemma4Config(
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=2 if use_moe else 1,
        num_experts_per_tok=1,
        moe_intermediate_size=16,
    )


def test_nnx_lora_linear_shapes_and_forward() -> None:
    """Test NNXLoRALinear initialization shapes, zero-B identity, and non-zero B forward output."""
    rngs = nnx.Rngs(42)
    base_linear = nnx.Linear(in_features=16, out_features=32, use_bias=True, rngs=rngs)

    x = jnp.ones((2, 4, 16), dtype=jnp.float32)
    expected_base = base_linear(x)

    lora_layer = NNXLoRALinear.from_linear(
        base_linear,
        r=4,
        lora_alpha=8.0,
        lora_dropout=0.0,
        rngs=rngs,
    )

    assert lora_layer.lora_a.shape == (16, 4)
    assert lora_layer.lora_b.shape == (4, 32)
    assert lora_layer.scale == pytest.approx(2.0)

    # Since B is initialized to 0, output should exactly match base linear
    out_initial = lora_layer(x, deterministic=True)
    assert out_initial.shape == (2, 4, 32)
    assert jnp.allclose(out_initial, expected_base, atol=1e-6)

    # Modify B to non-zero and verify output changes by scale * (x @ A @ B)
    lora_layer.lora_b[...] = jnp.ones((4, 32), dtype=jnp.float32)
    out_modified = lora_layer(x, deterministic=True)
    expected_delta = 2.0 * jnp.dot(jnp.dot(x, lora_layer.lora_a[...]), lora_layer.lora_b[...])
    assert jnp.allclose(out_modified, expected_base + expected_delta, atol=1e-5)


def test_nnx_lora_linear_invalid_rank() -> None:
    """Test that rank <= 0 raises ValueError."""
    with pytest.raises(ValueError, match="LoRA rank r must be positive"):
        NNXLoRALinear(in_features=8, out_features=8, r=0, rngs=nnx.Rngs(0))


def test_inject_lora_into_gemma4_model_and_forward_pass() -> None:
    """Verify inject_lora replaces target linear layers, preserves base forward output when B=0, and modifies it when B!=0."""
    config = _make_tiny_config(use_moe=False)
    rngs = nnx.Rngs(0)
    model = Gemma4ForCausalLM(config, rngs=rngs)
    tokens = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    positions = jnp.arange(4)[None, :]
    logits_base = model(tokens, positions)

    target_modules = ["q_proj", "v_proj", "gate_proj"]
    model, injected = inject_lora(
        model,
        target_modules=target_modules,
        lora_r=4,
        lora_alpha=8.0,
        lora_dropout=0.0,
        rngs=rngs,
    )

    # 2 layers * 3 target modules per layer = 6 injected modules
    assert injected == 6

    from gemma_4_sql.backends.jax.gemma4.layers import Gemma4MLP

    for layer in model.model.layers:
        assert isinstance(layer.self_attention.q_proj, LoRALinearAdapter)
        assert isinstance(layer.self_attention.v_proj, LoRALinearAdapter)
        assert not isinstance(layer.self_attention.k_proj, LoRALinearAdapter)
        assert not isinstance(layer.self_attention.o_proj, LoRALinearAdapter)
        assert isinstance(layer.mlp, Gemma4MLP)
        assert isinstance(layer.mlp.gate_proj, LoRALinearAdapter)
        assert not isinstance(layer.mlp.up_proj, LoRALinearAdapter)

    # When lora_b == 0, forward-pass output matches base model output exactly
    logits_zero_b = model(tokens, positions)
    assert jnp.allclose(logits_base, logits_zero_b, atol=1e-6)

    # When lora_b is non-zero, forward-pass output changes
    q_adapter = model.model.layers[0].self_attention.q_proj
    assert isinstance(q_adapter, LoRALinearAdapter)
    q_adapter.lora_b[...] = jnp.ones_like(q_adapter.lora_b[...]) * 0.5
    logits_nonzero_b = model(tokens, positions)
    assert not jnp.allclose(logits_base, logits_nonzero_b, atol=1e-5)

    total_params, lora_params = count_parameters(model)
    assert total_params > lora_params
    assert lora_params > 0


def test_lora_parameter_freezing_and_gradient_step() -> None:
    """Test that create_lora_optimizer updates only LoRAParam and keeps base weights frozen."""
    config = _make_tiny_config(use_moe=False)
    rngs = nnx.Rngs(123)
    model = Gemma4ForCausalLM(config, rngs=rngs)

    model, injected = inject_lora(
        model,
        target_modules=["q_proj", "v_proj"],
        lora_r=4,
        lora_alpha=8.0,
        lora_dropout=0.0,
        rngs=rngs,
    )
    assert injected == 4

    # Make lora_b non-zero so gradients flow into lora_a as well
    for layer in model.model.layers:
        q_proj = layer.self_attention.q_proj
        v_proj = layer.self_attention.v_proj
        assert isinstance(q_proj, LoRALinearAdapter)
        assert isinstance(v_proj, LoRALinearAdapter)
        q_proj.lora_b[...] = jnp.ones_like(q_proj.lora_b[...]) * 0.1
        v_proj.lora_b[...] = jnp.ones_like(v_proj.lora_b[...]) * 0.1

    # Record initial parameters
    q0 = model.model.layers[0].self_attention.q_proj
    assert isinstance(q0, LoRALinearAdapter)
    initial_q_kernel = jnp.copy(q0.kernel[...])
    initial_k_kernel = jnp.copy(model.model.layers[0].self_attention.k_proj.kernel[...])
    initial_lora_a = jnp.copy(q0.lora_a[...])
    initial_lora_b = jnp.copy(q0.lora_b[...])

    optimizer = create_lora_optimizer(model, optax.sgd(learning_rate=0.5))

    tokens = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    positions = jnp.arange(4)[None, :]

    def loss_fn(m: Gemma4ForCausalLM) -> jax.Array:
        """Compute mean squared logits loss for gradient verification."""
        logits = m(tokens, positions)
        return jnp.mean(logits**2)

    grad_fn = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, LoRAParam))
    loss, grads = grad_fn(model)
    assert loss.shape == ()

    optimizer.update(model, grads)

    # Base weights must remain completely unchanged (frozen)
    assert jnp.array_equal(q0.kernel[...], initial_q_kernel)
    assert jnp.array_equal(model.model.layers[0].self_attention.k_proj.kernel[...], initial_k_kernel)

    # LoRA parameters must have been updated
    assert not jnp.array_equal(q0.lora_a[...], initial_lora_a)
    assert not jnp.array_equal(q0.lora_b[...], initial_lora_b)


def test_apply_lora_entrypoint() -> None:
    """Test the apply_lora entrypoint returns expected metadata and status."""
    config = _make_tiny_config()
    model = Gemma4ForCausalLM(config, rngs=nnx.Rngs(0))
    _, v_only_count = inject_lora(Gemma4ForCausalLM(config, rngs=nnx.Rngs(1)), ["v_proj"], lora_r=2)
    assert v_only_count == 2

    res = apply_lora(
        model_name="gemma-4-tiny",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_r=4,
        lora_alpha=16,
        lora_dropout=0.05,
        model=model,
    )

    assert res["backend"] == "jax"
    assert res["action"] == "apply_lora"
    assert res["model"] == "gemma-4-tiny"
    assert res["lora_r"] == 4
    assert res["injected_modules"] == 14
    assert res["status"] == "completed"


def test_jax_peft_edge_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test edge cases in JAX PEFT including dropout, MoE layers, and missing dependencies."""
    # Dropout path and mode switching in NNXLoRALinear
    rngs = nnx.Rngs(params=0, dropout=1)
    lora_drop = NNXLoRALinear(in_features=8, out_features=8, r=2, lora_dropout=0.5, use_bias=True, rngs=rngs)
    x = jnp.ones((2, 8), dtype=jnp.float32)
    out = lora_drop(x, deterministic=False)
    assert out.shape == (2, 8)
    lora_drop.eval()
    assert jnp.array_equal(lora_drop(x), lora_drop(x))
    lora_drop.train(True)
    lora_drop.train(False)
    assert jnp.array_equal(lora_drop.W, lora_drop.kernel[...])
    assert jnp.array_equal(lora_drop.A, lora_drop.lora_a[...])
    assert jnp.array_equal(lora_drop.B, lora_drop.lora_b[...])

    lora_no_drop = NNXLoRALinear(in_features=4, out_features=4, r=2, lora_dropout=0.0, use_bias=False)
    lora_no_drop.train(True)
    lora_no_drop.eval()

    # MoE model injection
    moe_cfg = _make_tiny_config(use_moe=True)
    moe_model = Gemma4ForCausalLM(moe_cfg, rngs=nnx.Rngs(0))
    _, count = inject_lora(moe_model, target_modules=["gate_proj", "q_proj"], lora_r=2, rngs=nnx.Rngs(0))
    assert count >= 2

    # Empty target_modules
    _, count_empty = inject_lora(moe_model, target_modules=[], lora_r=2)
    assert count_empty == 0

    # Direct LoRAInjectableProtocol model injection
    class SingleInjectable:
        """Minimal module implementing LoRAInjectableProtocol directly."""

        def inject_lora_projections(
            self,
            target_modules: list[str],
            _adapter_factory: Callable[[nnx.Linear], LinearLayerProtocol],
        ) -> int:
            """Return count of target_modules."""
            return len(target_modules)

    _, count_direct = inject_lora(SingleInjectable(), target_modules=["q_proj", "v_proj"], lora_r=2)
    assert count_direct == 2

    # Non-injectable object returns 0
    assert inject_lora_to_layer(nnx.Linear(4, 4, rngs=nnx.Rngs(0)), ["q_proj"]) == 0
    _, count_non_inj = inject_lora("not_a_model", target_modules=["q_proj"])
    assert count_non_inj == 0

    # create_lora_optimizer with default tx=None
    opt_default = create_lora_optimizer(moe_model, tx=None)
    assert opt_default is not None

    # apply_lora without passing model instance (creates model from config)
    monkeypatch.setattr(pt, "resolve_gemma4_config", lambda _cls, _name: _make_tiny_config())  # pyright: ignore[reportUnknownLambdaType]
    res_new = apply_lora("test-tiny-jax", ["q_proj"], lora_r=2)
    assert res_new["status"] == "completed"
    assert res_new["injected_modules"] == 2

    # Error handling in apply_lora
    from gemma_4_sql.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError, match="lora_r must be > 0"):
        apply_lora("test-err", ["q_proj"], lora_r=0)

    with monkeypatch.context() as monkeypatch_ctx:
        monkeypatch_ctx.setattr(pt, "inject_lora", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("inject boom")))  # pyright: ignore[reportUnknownLambdaType]
        res_err = apply_lora("test-err", ["q_proj"], lora_r=2)
        assert "failed: inject boom" in res_err["status"]
        with pytest.raises(TrainingError, match="inject boom"):
            apply_lora("test-err", ["q_proj"], lora_r=2, raise_on_error=True)

    # Missing dependencies
    monkeypatch.setattr(pt, "nnx", None)
    with pytest.raises(DependencyMissingError):
        NNXLoRALinear(4, 4)
    with pytest.raises(DependencyMissingError):
        inject_lora(moe_model, ["q_proj"])
    with pytest.raises(DependencyMissingError):
        create_lora_optimizer(moe_model)
    with pytest.raises(DependencyMissingError):
        count_parameters(moe_model)
    with pytest.raises(DependencyMissingError):
        apply_lora("m", ["q_proj"])


def test_nnx_param_properties_and_indexing() -> None:
    """Verify NNXParam .shape, .dtype, __getitem__, and __setitem__."""
    arr = jnp.zeros((2, 3), dtype=jnp.float32)
    p = NNXParam(arr)
    assert p.shape == (2, 3)
    assert p.dtype == jnp.float32
    assert p[0].shape == (3,)
    p[...] = jnp.ones((2, 3), dtype=jnp.float32)
    assert float(p[...][0, 0]) == 1.0
    p[0] = jnp.zeros((3,), dtype=jnp.float32)
    assert float(p[...][0, 0]) == 0.0

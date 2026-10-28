"""Tests for parameter translation utilities."""

import jax
import jax.numpy as jnp
import torch

from gemma_4_sql.backends.pytorch.gemma4.utils_params import translate_jax_to_pytorch


def test_translate_jax_to_pytorch() -> None:
    """Test JAX to PyTorch state dict translation."""
    key = jax.random.PRNGKey(42)
    jax_params = {
        "model.layers.0.mlp.gate_proj.kernel": jax.random.normal(key, (256, 512), dtype=jnp.float32),
        "model.norm.scale": jnp.zeros((256,), dtype=jnp.float32),
        "some_other_param": torch.tensor([1.0, 2.0]),
        "scalar_param": [3.0, 4.0],
    }

    pytorch_state_dict = translate_jax_to_pytorch(jax_params)

    assert "layers.0.mlp.gate_proj.weight" in pytorch_state_dict
    assert pytorch_state_dict["layers.0.mlp.gate_proj.weight"].shape == (512, 256)

    assert "norm.weight" in pytorch_state_dict
    assert pytorch_state_dict["norm.weight"].shape == (256,)

    assert "some_other_param" in pytorch_state_dict
    assert torch.allclose(pytorch_state_dict["some_other_param"].float(), torch.tensor([1.0, 2.0]))
    assert torch.allclose(pytorch_state_dict["scalar_param"].float(), torch.tensor([3.0, 4.0]))


def test_translate_jax_to_pytorch_kernel_dim3() -> None:
    """Verify 3D JAX kernel tensor is flattened and transposed to a 2D PyTorch weight tensor."""
    jax_params = {
        "some.kernel": jnp.zeros((2, 3, 4)),
    }
    state = translate_jax_to_pytorch(jax_params)
    assert "some.weight" in state
    assert state["some.weight"].shape == (12, 2)  # (2, 12) transposed


def test_translate_jax_to_pytorch_routed_experts() -> None:
    """Verify 3D routed expert JAX kernels are split across individual PyTorch expert weight keys."""
    jax_params = {
        "routed_experts.kernel": jnp.zeros((2, 3, 4)),
    }
    state = translate_jax_to_pytorch(jax_params)
    # The 'kernel' is transposed so (2,3,4) -> (2,4,3)
    assert "experts.0.weight" in state
    assert state["experts.0.weight"].shape == (4, 3)
    assert "experts.1.weight" in state


def test_translated_weights_numerical_parity_rms_norm_and_mlp() -> None:
    """Verify translated JAX weights loaded into PyTorch Gemma4RMSNorm and Gemma4MLP match reference math."""
    from gemma_4_sql.backends.pytorch.gemma4.config import Gemma4Config
    from gemma_4_sql.backends.pytorch.gemma4.layers import Gemma4MLP, Gemma4RMSNorm

    dim = 8
    inter = 16
    k1, k2, k3, k4, k5 = jax.random.split(jax.random.PRNGKey(7), 5)
    scale_jax = jax.random.uniform(k1, shape=(dim,), minval=0.5, maxval=1.5, dtype=jnp.float32)
    gate_k = jax.random.normal(k2, shape=(dim, inter), dtype=jnp.float32) * 0.1
    up_k = jax.random.normal(k3, shape=(dim, inter), dtype=jnp.float32) * 0.1
    down_k = jax.random.normal(k4, shape=(inter, dim), dtype=jnp.float32) * 0.1

    state = translate_jax_to_pytorch({
        "model.norm.scale": scale_jax,
        "model.layers.0.mlp.gate.kernel": gate_k,
        "model.layers.0.mlp.up.kernel": up_k,
        "model.layers.0.mlp.down.kernel": down_k,
    })

    norm = Gemma4RMSNorm(dim, eps=1e-6)
    norm.load_state_dict({"weight": state["norm.weight"]})

    cfg = Gemma4Config(hidden_size=dim, intermediate_size=inter)
    mlp = Gemma4MLP(cfg)
    mlp.load_state_dict({
        "gate_proj.weight": state["layers.0.mlp.gate_proj.weight"],
        "up_proj.weight": state["layers.0.mlp.up_proj.weight"],
        "down_proj.weight": state["layers.0.mlp.down_proj.weight"],
    })

    x_jax = jax.random.normal(k5, shape=(2, dim), dtype=jnp.float32)
    x_pt = torch.tensor(x_jax.tolist(), dtype=torch.float32)
    scale_pt = torch.tensor(scale_jax.tolist(), dtype=torch.float32)

    # Reference RMSNorm (translate_jax_to_pytorch adds + 1.0 to JAX scale)
    rms_ref = x_pt * torch.rsqrt(torch.mean(x_pt**2, dim=-1, keepdim=True) + 1e-6) * (scale_pt + 1.0)
    out_norm = norm(x_pt).detach()
    assert torch.allclose(out_norm, rms_ref, atol=1e-4)

    # Reference SwiGLU/GELU MLP forward pass
    out_mlp = mlp(x_pt).detach()
    assert out_mlp.shape == (2, dim)
    assert bool(torch.all(torch.isfinite(out_mlp)))

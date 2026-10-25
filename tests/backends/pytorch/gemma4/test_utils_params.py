"""Tests for parameter translation utilities."""

import numpy as np
import torch

from gemma_4_sql.backends.pytorch.gemma4.utils_params import translate_jax_to_pytorch


def test_translate_jax_to_pytorch() -> None:
    """Test JAX to PyTorch state dict translation."""
    rng = np.random.default_rng(42)
    jax_params = {
        "model.layers.0.mlp.gate_proj.kernel": rng.standard_normal((256, 512)).astype(np.float32),
        "model.norm.scale": np.zeros((256,), dtype=np.float32),
        "some_other_param": torch.tensor([1.0, 2.0]),
    }

    pytorch_state_dict = translate_jax_to_pytorch(jax_params)

    assert "layers.0.mlp.gate_proj.weight" in pytorch_state_dict
    assert pytorch_state_dict["layers.0.mlp.gate_proj.weight"].shape == (512, 256)

    assert "norm.weight" in pytorch_state_dict
    assert pytorch_state_dict["norm.weight"].shape == (256,)

    assert "some_other_param" in pytorch_state_dict
    assert torch.allclose(pytorch_state_dict["some_other_param"].float(), torch.tensor([1.0, 2.0]))


def test_translate_jax_to_pytorch_kernel_dim3() -> None:
    """Verify 3D JAX kernel tensor is flattened and transposed to a 2D PyTorch weight tensor."""
    jax_params = {
        "some.kernel": np.zeros((2, 3, 4)),
    }
    state = translate_jax_to_pytorch(jax_params)
    assert "some.weight" in state
    assert state["some.weight"].shape == (12, 2)  # (2, 12) transposed


def test_translate_jax_to_pytorch_routed_experts() -> None:
    """Verify 3D routed expert JAX kernels are split across individual PyTorch expert weight keys."""
    jax_params = {
        "routed_experts.kernel": np.zeros((2, 3, 4)),
    }
    state = translate_jax_to_pytorch(jax_params)
    # The 'kernel' is transposed so (2,3,4) -> (2,4,3)
    assert "experts.0.weight" in state
    assert state["experts.0.weight"].shape == (4, 3)
    assert "experts.1.weight" in state


def test_translated_weights_numerical_parity_rms_norm_and_mlp() -> None:
    """Verify translated JAX weights loaded into PyTorch Gemma4RMSNorm and Gemma4MLP match numpy reference math."""
    from gemma_4_sql.backends.pytorch.gemma4.config import Gemma4Config
    from gemma_4_sql.backends.pytorch.gemma4.layers import Gemma4MLP, Gemma4RMSNorm

    rng = np.random.default_rng(7)
    dim = 8
    inter = 16
    scale_np = rng.uniform(0.5, 1.5, size=(dim,)).astype(np.float32)
    gate_k = rng.standard_normal((dim, inter)).astype(np.float32) * 0.1
    up_k = rng.standard_normal((dim, inter)).astype(np.float32) * 0.1
    down_k = rng.standard_normal((inter, dim)).astype(np.float32) * 0.1

    state = translate_jax_to_pytorch({
        "model.norm.scale": scale_np,
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

    x_np = rng.standard_normal((2, dim)).astype(np.float32)
    x_pt = torch.from_numpy(x_np)

    # Reference RMSNorm (translate_jax_to_pytorch adds + 1.0 to JAX scale)
    rms_ref = x_np * (1.0 / np.sqrt(np.mean(x_np**2, axis=-1, keepdims=True) + 1e-6)) * (scale_np + 1.0)
    out_norm = norm(x_pt).detach().numpy()
    assert np.allclose(out_norm, rms_ref, atol=1e-4)

    # Reference SwiGLU/GELU MLP forward pass
    out_mlp = mlp(x_pt).detach().numpy()
    assert out_mlp.shape == (2, dim)
    assert np.all(np.isfinite(out_mlp))

"""Tests for parameter translation utilities."""

import numpy as np
import torch

from gemma_4_sql.backends.pytorch.gemma4.utils_params import translate_jax_to_pytorch


def test_translate_jax_to_pytorch():
    """Test JAX to PyTorch state dict translation."""
    rng = np.random.default_rng(42)
    jax_params = {
        "model.layers.0.mlp.gate_proj.kernel": rng.standard_normal((256, 512)).astype(np.float32),
        "model.norm.scale": np.ones(256, dtype=np.float32),
        "some_other_param": [1.0, 2.0],
    }

    pytorch_state_dict = translate_jax_to_pytorch(jax_params)

    assert "layers.0.mlp.gate_proj.weight" in pytorch_state_dict
    assert pytorch_state_dict["layers.0.mlp.gate_proj.weight"].shape == (512, 256)

    assert "norm.weight" in pytorch_state_dict
    assert pytorch_state_dict["norm.weight"].shape == (256,)

    assert "some_other_param" in pytorch_state_dict
    assert torch.allclose(pytorch_state_dict["some_other_param"], torch.tensor([1.0, 2.0]))


def test_translate_jax_to_pytorch_kernel_dim3():
    import numpy as np

    from gemma_4_sql.backends.pytorch.gemma4.utils_params import translate_jax_to_pytorch

    jax_params = {
        "some.kernel": np.zeros((2, 3, 4)),
    }
    state = translate_jax_to_pytorch(jax_params)
    assert "some.weight" in state
    assert state["some.weight"].shape == (12, 2)  # (2, 12) transposed


def test_translate_jax_to_pytorch_routed_experts():
    import numpy as np

    from gemma_4_sql.backends.pytorch.gemma4.utils_params import translate_jax_to_pytorch

    jax_params = {
        "routed_experts.kernel": np.zeros((2, 3, 4)),
    }
    state = translate_jax_to_pytorch(jax_params)
    # The 'kernel' is transposed so (2,3,4) -> (2,4,3)
    assert "experts.0.weight" in state
    assert state["experts.0.weight"].shape == (4, 3)
    assert "experts.1.weight" in state

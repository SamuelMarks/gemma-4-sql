"""Tests for parameter translation utilities."""

import numpy as np
import torch

from gemma_4_sql.backends.pytorch.gemma4.utils_params import translate_jax_to_pytorch


def test_translate_jax_to_pytorch():
    """Test JAX to PyTorch state dict translation."""
    jax_params = {
        "model.layers.0.mlp.gate_proj.kernel": np.random.randn(256, 512).astype(np.float32),
        "model.norm.scale": np.ones(256, dtype=np.float32),
        "some_other_param": [1.0, 2.0],
    }

    pytorch_state_dict = translate_jax_to_pytorch(jax_params)

    assert "model.layers.0.mlp.gate_proj.weight" in pytorch_state_dict
    assert pytorch_state_dict["model.layers.0.mlp.gate_proj.weight"].shape == (512, 256)

    assert "model.norm.weight" in pytorch_state_dict
    assert pytorch_state_dict["model.norm.weight"].shape == (256,)

    assert "some_other_param" in pytorch_state_dict
    assert torch.allclose(pytorch_state_dict["some_other_param"], torch.tensor([1.0, 2.0]))

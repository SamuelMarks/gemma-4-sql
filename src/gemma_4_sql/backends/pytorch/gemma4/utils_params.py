"""Parameter translation utilities for JAX to PyTorch."""

from typing import Any

import torch


def translate_jax_to_pytorch(jax_params: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Translate JAX parameters to PyTorch state dict.

    This function handles the necessary transposition for Dense/Linear layers.

    Args:
        jax_params: Dictionary of JAX parameters.

    Returns:
        PyTorch state dictionary with mapped keys and weights.

    """
    pytorch_state_dict: dict[str, torch.Tensor] = {}

    for key, value in jax_params.items():
        if hasattr(value, "__array__"):
            import numpy as np

            tensor = torch.tensor(np.array(value).tolist())
        else:
            tensor = torch.tensor(value)

        key = key.replace("model.embed_tokens.embedding", "embed_tokens.weight")
        key = key.replace("model.layers.", "layers.")
        key = key.replace("model.norm.", "norm.")
        key = key.replace("vision_tower.", "vision_model.")
        key = key.replace("audio_tower.", "audio_model.")

        if "kernel" in key:
            if tensor.dim() > 2 and "routed_experts" not in key:
                tensor = tensor.view(tensor.size(0), -1)
            tensor = tensor.transpose(-2, -1)
            key = key.replace("kernel", "weight")

        if "scale" in key:
            key = key.replace("scale", "weight")
            tensor = tensor + 1.0

        key = key.replace(".mlp.gate.weight", ".mlp.gate_proj.weight")
        key = key.replace(".mlp.up.weight", ".mlp.up_proj.weight")
        key = key.replace(".mlp.down.weight", ".mlp.down_proj.weight")
        key = key.replace(".self_attention.", ".self_attn.")

        if "routed_experts" in key:
            # JAX shape: (num_experts, in_dim, out_dim)
            # PyTorch expects layers.0.mlp.experts.N.weight with shape (out_dim, in_dim)
            # The tensor is already transposed by the "kernel" check above -> (num_experts, out_dim, in_dim)
            base_key = key.replace("routed_experts.", "experts.")
            for i in range(tensor.size(0)):
                expert_key = base_key.replace("experts.", f"experts.{i}.")
                pytorch_state_dict[expert_key] = tensor[i]
        else:
            pytorch_state_dict[key] = tensor

    return pytorch_state_dict

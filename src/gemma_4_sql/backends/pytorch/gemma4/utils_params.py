"""Parameter translation utilities for converting JAX Gemma 4 weights to PyTorch."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def translate_jax_to_pytorch(jax_params: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Translate JAX parameters to a PyTorch `state_dict`.

    Handles zero-copy NumPy buffer conversion, kernel transposition for linear layers,
    RMSNorm scale offset adjustment (`+ 1.0`), and routed MoE expert weight splitting.

    Args:
        jax_params: Dictionary of JAX parameter arrays keyed by dotted path.

    Returns:
        PyTorch state dictionary with mapped keys and `torch.Tensor` weights.

    """
    pytorch_state_dict: dict[str, torch.Tensor] = {}

    for key, value in jax_params.items():
        tensor = value.clone() if isinstance(value, torch.Tensor) else torch.from_numpy(np.asarray(value).copy())

        key = key.replace("model.embed_tokens.embedding", "embed_tokens.weight")
        key = key.replace("model.layers.", "layers.")
        key = key.replace("model.norm.", "norm.")
        key = key.replace("vision_tower.", "vision_model.")
        key = key.replace("audio_tower.", "audio_model.")

        if "kernel" in key:
            if tensor.dim() > 2 and "routed_experts" not in key:
                tensor = tensor.reshape(tensor.size(0), -1)
            tensor = tensor.transpose(-2, -1).contiguous()
            key = key.replace("kernel", "weight")

        if "scale" in key:
            key = key.replace("scale", "weight")
            tensor = tensor + 1.0

        key = key.replace(".mlp.gate.weight", ".mlp.gate_proj.weight")
        key = key.replace(".mlp.up.weight", ".mlp.up_proj.weight")
        key = key.replace(".mlp.down.weight", ".mlp.down_proj.weight")
        key = key.replace(".self_attention.", ".self_attn.")

        if "routed_experts" in key:
            base_key = key.replace("routed_experts.", "experts.")
            for i in range(tensor.size(0)):
                expert_key = base_key.replace("experts.", f"experts.{i}.")
                pytorch_state_dict[expert_key] = tensor[i]
        else:
            pytorch_state_dict[key] = tensor

    return pytorch_state_dict

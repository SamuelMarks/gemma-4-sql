"""Parameter translation utilities for converting JAX Gemma 4 weights to PyTorch."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from gemma_4_sql.type_hints import TensorType, ToListArrayProtocol


def _map_jax_key_and_tensor(raw_key: str, tensor: torch.Tensor) -> tuple[str, torch.Tensor]:
    """Map a single JAX parameter path and tensor into its PyTorch key and layout.

    Args:
        raw_key: Dotted parameter path from a Flax NNX Gemma 4 state dictionary.
        tensor: Cloned PyTorch tensor extracted from the JAX array.

    Returns:
        Tuple of `(mapped_pytorch_key, transformed_tensor)`.

    """
    key = raw_key
    key = key.replace("model.embed_tokens.embedding", "embed_tokens.weight")
    key = key.replace("model.embed_tokens_per_layer.embedding", "embed_tokens_per_layer.weight")
    key = key.replace("model.per_layer_model_projection.", "per_layer_model_projection.")
    key = key.replace("model.per_layer_projection_norm.", "per_layer_projection_norm.")
    key = key.replace("model.layers.", "layers.")
    key = key.replace("model.norm.", "norm.")
    key = key.replace("vision_tower.", "vision_model.")
    key = key.replace("audio_tower.", "audio_model.")

    key = key.replace(".pre_self_attention_norm.", ".input_layernorm.")
    key = key.replace(".post_self_attention_norm.", ".post_attention_layernorm.")
    key = key.replace(".pre_ffw_norm.", ".pre_feedforward_layernorm.")
    key = key.replace(".post_ffw_norm.", ".post_feedforward_layernorm.")
    key = key.replace(".self_attention.", ".self_attn.")
    if key.startswith("self_attention."):
        key = f"self_attn.{key[len('self_attention.') :]}"

    # MoE router parameter mappings (preserve scale values without +1.0 RMSNorm offset)
    if key.endswith(".mlp.pre_forward_scale_2"):
        return (key.replace(".mlp.pre_forward_scale_2", ".mlp.router.scale"), tensor)
    if key.endswith(".mlp.per_expert_scale"):
        return (key.replace(".mlp.per_expert_scale", ".mlp.router.per_expert_scale"), tensor)
    was_mlp_gate_kernel = key.endswith(".mlp.gate.kernel")

    if "routed_experts." in key and (key.endswith("_kernel") or "kernel" in key):
        tensor = tensor.transpose(-2, -1).contiguous()
        key = f"{key[: -len('_kernel')]}.weight" if key.endswith("_kernel") else key.replace("kernel", "weight")
    elif "kernel" in key:
        if tensor.dim() > 2:
            tensor = tensor.reshape(-1, tensor.size(-1)) if "o_proj." in key else tensor.reshape(tensor.size(0), -1)
        tensor = tensor.transpose(-2, -1).contiguous()
        key = key.replace("kernel", "weight")

    if key.endswith(".scale") or key == "scale":
        key = "weight" if key == "scale" else f"{key[: -len('.scale')]}.weight"
        tensor = tensor + 1.0

    if not was_mlp_gate_kernel:
        key = key.replace(".mlp.gate.weight", ".mlp.gate_proj.weight")
    key = key.replace(".mlp.up.weight", ".mlp.up_proj.weight")
    key = key.replace(".mlp.down.weight", ".mlp.down_proj.weight")
    key = key.replace(".shared_experts.gate.weight", ".shared_experts.gate_proj.weight")
    key = key.replace(".shared_experts.up.weight", ".shared_experts.up_proj.weight")
    key = key.replace(".shared_experts.down.weight", ".shared_experts.down_proj.weight")

    return (key, tensor)


def translate_jax_to_pytorch(
    jax_params: Mapping[str, TensorType | torch.Tensor | ToListArrayProtocol | Sequence[float]],
) -> dict[str, torch.Tensor]:
    """Translate JAX parameters to a PyTorch `state_dict`.

    Handles tensor conversion, kernel transposition for linear layers,
    RMSNorm scale offset adjustment (`+ 1.0`), MoE router scale/gate remapping,
    and routed MoE expert weight splitting.

    Args:
        jax_params: Dictionary of JAX parameter arrays keyed by dotted path.

    Returns:
        PyTorch state dictionary with mapped keys and `torch.Tensor` weights.

    """
    pytorch_state_dict: dict[str, torch.Tensor] = {}
    moe_prefixes = {
        raw_k[: -len(".mlp.per_expert_scale")] for raw_k in jax_params if raw_k.endswith(".mlp.per_expert_scale")
    }

    for raw_key, value in jax_params.items():
        if isinstance(value, torch.Tensor):
            tensor = value.clone()
        elif isinstance(value, ToListArrayProtocol):
            tensor = torch.tensor(value.tolist(), dtype=torch.float32)
        else:
            tensor = torch.as_tensor(value, dtype=torch.float32).clone()

        key, tensor = _map_jax_key_and_tensor(raw_key, tensor)
        if raw_key.endswith(".mlp.gate.kernel"):
            prefix = raw_key[: -len(".mlp.gate.kernel")]
            if prefix in moe_prefixes:
                key = key.replace(".mlp.gate.weight", ".mlp.router.gate.weight")
            else:
                key = key.replace(".mlp.gate.weight", ".mlp.gate_proj.weight")

        if "routed_experts." in key:
            base_key = key.replace("routed_experts.", "experts.")
            for i in range(tensor.size(0)):
                expert_key = base_key.replace("experts.", f"experts.{i}.")
                pytorch_state_dict[expert_key] = tensor[i]
        else:
            pytorch_state_dict[key] = tensor

    return pytorch_state_dict

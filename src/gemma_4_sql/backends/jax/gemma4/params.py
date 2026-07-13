"""Parameter helpers for gemma4.

Provides parameter matching and checkpoint utilities.
"""

import logging
import re

import jax
import jax.numpy as jnp
import safetensors.flax as safetensors
from etils import epath
from flax import nnx

from gemma_4_sql.type_hints import JSONDict

from . import modeling as model_lib
from .utils_params import assign_weights_from_eval_shape, map_to_jax_key, stoi

logger = logging.getLogger(__name__)


def _get_text_mappings(transform_cls: type) -> dict[str, tuple[str, object]]:
    """Return text-specific safetensors mapping.

    Args:
        transform_cls: The transform cls.

    Returns:
        A tuple containing the results.
    """
    return {
        "^model\\.embed_tokens\\.weight$": ("model\\.embed_tokens\\.embedding", transform_cls.EMBED),
        "^model\\.embed_tokens_per_layer\\.weight$": ("model\\.embed_tokens_per_layer\\.embedding", transform_cls.EMBED),
        "^model\\.per_layer_model_projection\\.weight$": ("model\\.per_layer_model_projection\\.kernel", transform_cls.LINEAR),
        "^model\\.per_layer_projection_norm\\.weight$": ("model\\.per_layer_projection_norm\\.scale", transform_cls.DEFAULT),
        "^model\\.layers\\.(\\d+)\\.per_layer_input_gate\\.weight$": ("model\\.layers\\.\\1\\.per_layer_input_gate\\.kernel", transform_cls.LINEAR),
        "^model\\.layers\\.(\\d+)\\.per_layer_projection\\.weight$": ("model\\.layers\\.\\1\\.per_layer_projection\\.kernel", transform_cls.LINEAR),
        "^model\\.layers\\.(\\d+)\\.post_per_layer_input_norm\\.weight$": ("model\\.layers\\.\\1\\.post_per_layer_input_norm\\.scale", transform_cls.DEFAULT),
        "^model\\.layers\\.(\\d+)\\.input_layernorm\\.weight$": ("model\\.layers\\.\\1\\.pre_self_attention_norm\\.scale", transform_cls.DEFAULT),
        "^model\\.layers\\.(\\d+)\\.post_attention_layernorm\\.weight$": ("model\\.layers\\.\\1\\.post_self_attention_norm\\.scale", transform_cls.DEFAULT),
        "^model\\.layers\\.(\\d+)\\.pre_feedforward_layernorm\\.weight$": ("model\\.layers\\.\\1\\.pre_ffw_norm\\.scale", transform_cls.DEFAULT),
        "^model\\.layers\\.(\\d+)\\.post_feedforward_layernorm\\.weight$": ("model\\.layers\\.\\1\\.post_ffw_norm\\.scale", transform_cls.DEFAULT),
        "^model\\.layers\\.(\\d+)\\.self_attn\\.q_norm\\.weight$": ("model\\.layers\\.\\1\\.self_attention\\.q_norm\\.scale", transform_cls.DEFAULT),
        "^model\\.layers\\.(\\d+)\\.self_attn\\.k_norm\\.weight$": ("model\\.layers\\.\\1\\.self_attention\\.k_norm\\.scale", transform_cls.DEFAULT),
        "^model\\.layers\\.(\\d+)\\.self_attn\\.v_norm\\.weight$": ("model\\.layers\\.\\1\\.self_attention\\.v_norm\\.scale", transform_cls.DEFAULT),
        "^model\\.layers\\.(\\d+)\\.self_attn\\.q_proj\\.weight$": ("model\\.layers\\.\\1\\.self_attention\\.q_proj\\.kernel", transform_cls.LINEAR),
        "^model\\.layers\\.(\\d+)\\.self_attn\\.k_proj\\.weight$": ("model\\.layers\\.\\1\\.self_attention\\.k_proj\\.kernel", transform_cls.LINEAR),
        "^model\\.layers\\.(\\d+)\\.self_attn\\.v_proj\\.weight$": ("model\\.layers\\.\\1\\.self_attention\\.v_proj\\.kernel", transform_cls.LINEAR),
        "^model\\.layers\\.(\\d+)\\.self_attn\\.o_proj\\.weight$": ("model\\.layers\\.\\1\\.self_attention\\.o_proj\\.kernel", transform_cls.LINEAR),
        "^model\\.layers\\.(\\d+)\\.mlp\\.gate_proj\\.weight$": ("model\\.layers\\.\\1\\.mlp\\.gate_proj\\.kernel", transform_cls.LINEAR),
        "^model\\.layers\\.(\\d+)\\.mlp\\.up_proj\\.weight$": ("model\\.layers\\.\\1\\.mlp\\.up_proj\\.kernel", transform_cls.LINEAR),
        "^model\\.layers\\.(\\d+)\\.mlp\\.down_proj\\.weight$": ("model\\.layers\\.\\1\\.mlp\\.down_proj\\.kernel", transform_cls.LINEAR),
        "^model\\.layers\\.(\\d+)\\.block_sparse_moe\\.gate\\.weight$": ("model\\.layers\\.\\1\\.mlp\\.gate\\.kernel", transform_cls.LINEAR),
        "^model\\.layers\\.(\\d+)\\.mlp\\.routed_experts\\.gate_proj\\.weight$": ("model\\.layers\\.\\1\\.mlp\\.routed_experts\\.gate_proj_kernel", transform_cls.LINEAR_3D),
        "^model\\.layers\\.(\\d+)\\.mlp\\.routed_experts\\.up_proj\\.weight$": ("model\\.layers\\.\\1\\.mlp\\.routed_experts\\.up_proj_kernel", transform_cls.LINEAR_3D),
        "^model\\.layers\\.(\\d+)\\.mlp\\.routed_experts\\.down_proj\\.weight$": ("model\\.layers\\.\\1\\.mlp\\.routed_experts\\.down_proj_kernel", transform_cls.LINEAR_3D),
        "^model\\.layers\\.(\\d+)\\.shared_expert\\.gate_proj\\.weight$": ("model\\.layers\\.\\1\\.mlp\\.shared_experts\\.gate_proj\\.kernel", transform_cls.LINEAR),
        "^model\\.layers\\.(\\d+)\\.shared_expert\\.up_proj\\.weight$": ("model\\.layers\\.\\1\\.mlp\\.shared_experts\\.up_proj\\.kernel", transform_cls.LINEAR),
        "^model\\.layers\\.(\\d+)\\.shared_expert\\.down_proj\\.weight$": ("model\\.layers\\.\\1\\.mlp\\.shared_experts\\.down_proj\\.kernel", transform_cls.LINEAR),
        "^model\\.layers\\.(\\d+)\\.block_sparse_moe\\.pre_forward_scale_2\\.weight$": ("model\\.layers\\.\\1\\.mlp\\.pre_forward_scale_2", transform_cls.DEFAULT),
        "^model\\.layers\\.(\\d+)\\.block_sparse_moe\\.pre_feedforward_layernorm_2\\.weight$": ("model\\.layers\\.\\1\\.mlp\\.pre_feedforward_layernorm_2\\.scale", transform_cls.DEFAULT),
        "^model\\.layers\\.(\\d+)\\.block_sparse_moe\\.post_feedforward_layernorm_1\\.weight$": ("model\\.layers\\.\\1\\.mlp\\.post_feedforward_layernorm_1\\.scale", transform_cls.DEFAULT),
        "^model\\.layers\\.(\\d+)\\.block_sparse_moe\\.post_feedforward_layernorm_2\\.weight$": ("model\\.layers\\.\\1\\.mlp\\.post_feedforward_layernorm_2\\.scale", transform_cls.DEFAULT),
        "^model\\.layers\\.(\\d+)\\.block_sparse_moe\\.per_expert_scale$": ("model\\.layers\\.\\1\\.mlp\\.per_expert_scale", transform_cls.DEFAULT),
        "^model\\.layers\\.(\\d+)\\.layer_scalar\\.weight$": ("model\\.layers\\.\\1\\.layer_scalar", transform_cls.DEFAULT),
        "^model\\.norm\\.weight$": ("model\\.norm\\.scale", transform_cls.DEFAULT),
        "^lm_head\\.weight$": ("lm_head\\.kernel", transform_cls.LINEAR),
        "^embed_audio\\.embedding_projection\\.weight$": ("embed_audio\\.embedding_projection\\.kernel", transform_cls.LINEAR),
        "^multi_modal_projector\\.mm_input_projection_weight$": ("multi_modal_projector\\.mm_input_projection_weight", transform_cls.DEFAULT),
        "^multi_modal_projector\\.mm_soft_emb_norm\\.weight$": ("multi_modal_projector\\.mm_soft_emb_norm\\.scale", transform_cls.DEFAULT),
    }


def _get_audio_mappings(transform_cls: type) -> dict[str, tuple[str, object]]:
    """Return audio-specific safetensors mapping.

    Args:
        transform_cls: The transform cls.

    Returns:
        A tuple containing the results.
    """
    return {
        "^audio_tower\\.subsample_conv_projection\\.layer(\\d+)\\.conv\\.weight$": ("audio_tower\\.subsample_conv_projection\\.layer\\1\\.conv\\.kernel", transform_cls.CONV2D),
        "^audio_tower\\.subsample_conv_projection\\.layer(\\d+)\\.norm\\.weight$": ("audio_tower\\.subsample_conv_projection\\.layer\\1\\.norm\\.scale", transform_cls.DEFAULT),
        "^audio_tower\\.subsample_conv_projection\\.layer(\\d+)\\.norm\\.bias$": ("audio_tower\\.subsample_conv_projection\\.layer\\1\\.norm\\.bias", transform_cls.BIAS),
        "^audio_tower\\.subsample_conv_projection\\.input_proj_linear\\.weight$": ("audio_tower\\.subsample_conv_projection\\.input_proj_linear\\.kernel", transform_cls.LINEAR),
        "^audio_tower\\.layers\\.(\\d+)\\.feed_forward(\\d+)\\.ffw_layer_(\\d+)\\.linear\\.weight$": ("audio_tower\\.layers\\.\\1\\.feed_forward\\2\\.ffw_layer_\\3\\.linear\\.kernel", transform_cls.LINEAR),
        "^audio_tower\\.layers\\.(\\d+)\\.feed_forward(\\d+)\\.(pre|post)_layer_norm\\.weight$": ("audio_tower\\.layers\\.\\1\\.feed_forward\\2\\.\\3_layer_norm\\.scale", transform_cls.DEFAULT),
        "^audio_tower\\.layers\\.(\\d+)\\.self_attn\\.(q_proj|k_proj|v_proj|post)\\.linear\\.weight$": ("audio_tower\\.layers\\.\\1\\.self_attn\\.\\2\\.linear\\.kernel", transform_cls.LINEAR),
        "^audio_tower\\.layers\\.(\\d+)\\.self_attn\\.relative_k_proj\\.weight$": ("audio_tower\\.layers\\.\\1\\.self_attn\\.relative_k_proj\\.kernel", transform_cls.LINEAR),
        "^audio_tower\\.layers\\.(\\d+)\\.self_attn\\.per_dim_scale$": ("audio_tower\\.layers\\.\\1\\.self_attn\\.per_dim_scale", transform_cls.DEFAULT),
        "^audio_tower\\.layers\\.(\\d+)\\.lconv1d\\.(linear_start|linear_end)\\.linear\\.weight$": ("audio_tower\\.layers\\.\\1\\.lconv1d\\.\\2\\.linear\\.kernel", transform_cls.LINEAR),
        "^audio_tower\\.layers\\.(\\d+)\\.lconv1d\\.depthwise_conv1d\\.weight$": ("audio_tower\\.layers\\.\\1\\.lconv1d\\.depthwise_conv1d\\.conv\\.kernel", ((2, 1, 0), None, False)),
        "^audio_tower\\.layers\\.(\\d+)\\.lconv1d\\.(pre_layer_norm|conv_norm)\\.weight$": ("audio_tower\\.layers\\.\\1\\.lconv1d\\.\\2\\.scale", transform_cls.DEFAULT),
        "^audio_tower\\.layers\\.(\\d+)\\.norm_(pre_attn|post_attn|out)\\.weight$": ("audio_tower\\.layers\\.\\1\\.norm_\\2\\.scale", transform_cls.DEFAULT),
        "^audio_tower\\.output_proj\\.weight$": ("audio_tower\\.output_proj\\.kernel", transform_cls.LINEAR),
        "^audio_tower\\.output_proj\\.bias$": ("audio_tower\\.output_proj\\.bias", transform_cls.BIAS),
    }


def _get_vision_mappings(transform_cls: type) -> dict[str, tuple[str, object]]:
    """Return vision-specific safetensors mapping.

    Args:
        transform_cls: The transform cls.

    Returns:
        A tuple containing the results.
    """
    return {
        "^vision_tower\\.vision_model\\.embeddings\\.patch_embedding\\.bias$": ("vision_tower\\.embeddings\\.patch_embedding\\.bias", transform_cls.BIAS),
        "^vision_tower\\.vision_model\\.embeddings\\.patch_embedding\\.weight$": ("vision_tower\\.embeddings\\.patch_embedding\\.kernel", transform_cls.CONV2D),
        "^vision_tower\\.vision_model\\.embeddings\\.position_embedding\\.weight$": ("vision_tower\\.embeddings\\.position_embedding\\.embedding", transform_cls.EMBED),
        "^vision_tower\\.vision_model\\.encoder\\.layers\\.(\\d+)\\.layer_norm(\\d+)\\.weight$": ("vision_tower\\.layers\\.\\1\\.layer_norm\\2\\.scale", transform_cls.DEFAULT),
        "^vision_tower\\.vision_model\\.encoder\\.layers\\.(\\d+)\\.mlp\\.fc(\\d+)\\.bias$": ("vision_tower\\.layers\\.\\1\\.mlp\\.fc\\2\\.bias", transform_cls.BIAS),
        "^vision_tower\\.vision_model\\.encoder\\.layers\\.(\\d+)\\.mlp\\.fc(\\d+)\\.weight$": ("vision_tower\\.layers\\.\\1\\.mlp\\.fc\\2\\.kernel", transform_cls.LINEAR),
        "^vision_tower\\.vision_model\\.encoder\\.layers\\.(\\d+)\\.self_attn\\.k_proj\\.bias$": ("vision_tower\\.layers\\.\\1\\.self_attn\\.k_proj\\.bias", transform_cls.BIAS),
        "^vision_tower\\.vision_model\\.encoder\\.layers\\.(\\d+)\\.self_attn\\.k_proj\\.weight$": ("vision_tower\\.layers\\.\\1\\.self_attn\\.k_proj\\.kernel", transform_cls.LINEAR),
        "^vision_tower\\.vision_model\\.encoder\\.layers\\.(\\d+)\\.self_attn\\.out_proj\\.bias$": ("vision_tower\\.layers\\.\\1\\.self_attn\\.out_proj\\.bias", transform_cls.BIAS),
        "^vision_tower\\.vision_model\\.encoder\\.layers\\.(\\d+)\\.self_attn\\.out_proj\\.weight$": ("vision_tower\\.layers\\.\\1\\.self_attn\\.out_proj\\.kernel", transform_cls.LINEAR),
        "^vision_tower\\.vision_model\\.encoder\\.layers\\.(\\d+)\\.self_attn\\.q_proj\\.bias$": ("vision_tower\\.layers\\.\\1\\.self_attn\\.q_proj\\.bias", transform_cls.BIAS),
        "^vision_tower\\.vision_model\\.encoder\\.layers\\.(\\d+)\\.self_attn\\.q_proj\\.weight$": ("vision_tower\\.layers\\.\\1\\.self_attn\\.q_proj\\.kernel", transform_cls.LINEAR),
        "^vision_tower\\.vision_model\\.encoder\\.layers\\.(\\d+)\\.self_attn\\.v_proj\\.bias$": ("vision_tower\\.layers\\.\\1\\.self_attn\\.v_proj\\.bias", transform_cls.BIAS),
        "^vision_tower\\.vision_model\\.encoder\\.layers\\.(\\d+)\\.self_attn\\.v_proj\\.weight$": ("vision_tower\\.layers\\.\\1\\.self_attn\\.v_proj\\.kernel", transform_cls.LINEAR),
        "^vision_tower\\.vision_model\\.post_layernorm\\.weight$": ("vision_tower\\.post_layernorm\\.scale", transform_cls.DEFAULT),
    }


def _get_key_and_transform_mapping() -> object:
    """Return the mapping from safetensors keys to JAX model keys and their transforms.

    Returns
    -------
        dict: A dictionary mapping safetensors regex patterns to a tuple containing the
        corresponding JAX model key pattern and the required Transform enum.

    """

    class Transform:
        """Specifies default transformation types for model parameter names."""

        DEFAULT = None
        BIAS = None
        LINEAR = ((1, 0), None, False)
        CONV2D = ((2, 3, 1, 0), None, False)
        EMBED = None
        LINEAR_3D = ((0, 2, 1), None, False)

    mapping = {}
    mapping.update(_get_text_mappings(Transform))
    mapping.update(_get_audio_mappings(Transform))
    mapping.update(_get_vision_mappings(Transform))
    return mapping


def _process_moe_tensor(match: re.Match[str], sf: object, torch_key: str, expert_tensors: dict[int, dict[str, dict[int, jax.Array]]]) -> None:
    """Process an MoE expert tensor."""
    (l_idx_str, e_idx_str, proj_type) = match.groups()
    (l_idx, e_idx) = (int(l_idx_str), int(e_idx_str))
    if l_idx not in expert_tensors:
        expert_tensors[l_idx] = {}
    if proj_type not in expert_tensors[l_idx]:
        expert_tensors[l_idx][proj_type] = {}
    expert_tensors[l_idx][proj_type][e_idx] = jnp.array(sf.get_tensor(torch_key))


def process_standard_tensor(sf: object, torch_key: str, jax_state: JSONDict, mapping: dict[str, tuple]) -> None:
    """Process a standard tensor.

    Raises:
        AttributeError: If the operation encounters an unexpected AttributeError.

    Raises:
        ImportError: If the operation encounters an unexpected ImportError.

    Raises:
        OSError: If the operation encounters an unexpected OSError.

    Raises:
        RuntimeError: If the operation encounters an unexpected RuntimeError.

    Raises:
        AttributeError: If the operation encounters an unexpected AttributeError.
        RuntimeError: If the operation encounters an unexpected RuntimeError.
        TypeError: If the operation encounters an unexpected TypeError.
        OSError: If the operation encounters an unexpected OSError.
        ImportError: If the operation encounters an unexpected ImportError.

    """
    tensor = jnp.array(sf.get_tensor(torch_key))
    (jax_key, transform) = map_to_jax_key(mapping, torch_key)
    if jax_key is None:
        return
    keys = [stoi(k) for k in jax_key.split("\\.")]
    try:
        assign_weights_from_eval_shape(keys, tensor, jax_state, torch_key, transform.value if hasattr(transform, "value") else transform)
    except (KeyError, ValueError) as e:
        logger.debug("Skipping assignment for %s: %s", torch_key, e)
    except (TypeError, AttributeError, ImportError, RuntimeError, OSError):
        raise


def _stack_and_assign_expert_tensors(expert_tensors: dict[int, dict[str, dict[int, jax.Array]]], mapping: dict[str, tuple[str, object]], jax_state: JSONDict) -> None:
    """Stack expert tensors and assign them to the jax state."""
    for l_idx, projs in expert_tensors.items():
        for proj_type, e_dict in projs.items():
            tensors = [e_dict[i] for i in sorted(e_dict.keys())]
            stacked = jnp.stack(tensors, axis=0)
            st_key = f"model.layers.{l_idx}.mlp.routed_experts.{proj_type}.weight"
            (jax_key, transform) = map_to_jax_key(mapping, st_key)
            if jax_key is not None:  # pragma: no cover
                keys = [stoi(k) for k in jax_key.split("\\.")]
                assign_weights_from_eval_shape(keys, stacked, jax_state, st_key, transform)


def _process_safetensors_file(f: object, moe_pattern: re.Pattern[str], expert_tensors: dict[int, dict[str, dict[int, jax.Array]]], jax_state: JSONDict, mapping: dict[str, tuple[str, object]]) -> None:
    """Process a single safetensors file."""
    with safetensors.safe_open(f, framework="numpy") as sf:
        for torch_key in list(sf.keys()):
            match = moe_pattern.match(torch_key)
            if match:
                _process_moe_tensor(match, sf, torch_key, expert_tensors)
            else:
                process_standard_tensor(sf, torch_key, jax_state, mapping)


def _fix_jax_state_embeddings(jax_state: JSONDict, gemma4: object, cfg: model_lib.ModelConfig) -> None:
    """Fix uninitialized state embeddings that evaluation shape might leave empty."""
    embed_scale = jax_state.get("model", {}).get("embed_scale")
    if embed_scale is not None and isinstance(embed_scale, getattr(jax, "ShapeDtypeStruct", type(None))):
        jax_state["model"]["embed_scale"] = jnp.array(cfg.hidden_size**0.5, dtype=jnp.bfloat16).astype(jnp.float32)

    if cfg.vision_config:
        pos_ids = jax_state.get("vision_tower", {}).get("embeddings", {}).get("position_ids")
        if pos_ids is not None and isinstance(pos_ids, jax.ShapeDtypeStruct):  # pragma: no cover
            jax_state["vision_tower"]["embeddings"]["position_ids"] = jnp.expand_dims(jnp.arange(gemma4.vision_tower.embeddings.num_patches), 0)


def create_gemma4_from_pretrained(file_dir: str, cfg: model_lib.ModelConfig) -> object:
    """Load safetensor weights from a file, then convert & merge into a flax.nnx model.

    Returns:
        object: The resulting output from the operation.

    Raises:
        ValueError: If the operation encounters an unexpected ValueError.

    """
    gc = __import__("gc")
    files = list(epath.Path(file_dir).expanduser().glob("*.safetensors"))
    if not files:
        msg = f"No safetensors found in {file_dir}"
        raise ValueError(msg)
    gemma4 = nnx.eval_shape(lambda: model_lib.Gemma4ForCausalLM(cfg, rngs=nnx.Rngs(0)))
    (graph_def, abs_state) = nnx.split(gemma4)
    jax_state = dict(abs_state.to_pure_dict() if hasattr(abs_state, "to_pure_dict") else getattr(abs_state, "to_flat_dict", lambda: dict(abs_state))())
    mapping = _get_key_and_transform_mapping()
    moe_pattern = re.compile("^model\\.layers\\.(\\d+)\\.block_sparse_moe\\.experts\\.(\\d+)\\.(gate_proj|up_proj|down_proj)\\.weight$")
    expert_tensors: dict[int, dict[str, dict[int, jax.Array]]] = {}
    for f in files:
        _process_safetensors_file(f, moe_pattern, expert_tensors, jax_state, mapping)
        gc.collect()
    _stack_and_assign_expert_tensors(expert_tensors, mapping, jax_state)
    _fix_jax_state_embeddings(jax_state, gemma4, cfg)
    if hasattr(nnx, "State"):
        return nnx.merge(graph_def, abs_state)
    return nnx.merge(graph_def, jax_state)

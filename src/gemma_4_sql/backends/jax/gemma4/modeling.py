"""Gemma 4 model implementation in JAX/Flax NNX."""

from __future__ import annotations

import inspect
import math
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from .attention import Gemma4Attention
from .audio import (
    Gemma4AudioAttention,
    Gemma4AudioCausalConv1d,
    Gemma4AudioFeedForward,
    Gemma4AudioLayer,
    Gemma4AudioLightConv1d,
    Gemma4AudioModel,
    Gemma4AudioRelPositionalEncoding,
    Gemma4AudioSubSampleConvProjection,
    Gemma4AudioSubSampleConvProjectionLayer,
    _compute_audio_attention_outputs,
    _convert_to_block,
    _extract_block_context,
    _rel_shift,
)
from .config import AttentionType, AudioConfig, ModelConfig, ModelConfigPresets, ShardConfig, ShardMode, VisionConfig, VisionShardConfig
from .decoder_layer import Gemma4DecoderLayer
from .layers import ConstVar, Gemma4ClippableLinear, Gemma4MLP, Gemma4RMSNorm, StatVar, _make_embed, _make_linear
from .moe import Gemma4MoE, Gemma4RoutedExperts
from .vision import Gemma4MultimodalEmbedder, Gemma4MultiModalProjector, SiglipAttention, SiglipEncoderLayer, SiglipMLP, SiglipVisionEmbeddings, SiglipVisionTransformer, _avg_pool_vision_outputs

__all__ = [
    "AttentionType",
    "AudioConfig",
    "ConstVar",
    "Gemma4Attention",
    "Gemma4AudioAttention",
    "Gemma4AudioCausalConv1d",
    "Gemma4AudioFeedForward",
    "Gemma4AudioLayer",
    "Gemma4AudioLightConv1d",
    "Gemma4AudioModel",
    "Gemma4AudioRelPositionalEncoding",
    "Gemma4AudioSubSampleConvProjection",
    "Gemma4AudioSubSampleConvProjectionLayer",
    "Gemma4ClippableLinear",
    "Gemma4DecoderLayer",
    "Gemma4ForCausalLM",
    "Gemma4MLP",
    "Gemma4MoE",
    "Gemma4Model",
    "Gemma4MultiModalProjector",
    "Gemma4MultimodalEmbedder",
    "Gemma4RMSNorm",
    "Gemma4RoutedExperts",
    "ModelConfig",
    "ModelConfigPresets",
    "ShardConfig",
    "ShardMode",
    "SiglipAttention",
    "SiglipEncoderLayer",
    "SiglipMLP",
    "SiglipVisionEmbeddings",
    "SiglipVisionTransformer",
    "StatVar",
    "VisionConfig",
    "VisionShardConfig",
    "_avg_pool_vision_outputs",
    "_compute_audio_attention_outputs",
    "_convert_to_block",
    "_extract_block_context",
    "_make_embed",
    "_make_linear",
    "_rel_shift",
]
if TYPE_CHECKING:
    from jax.sharding import PartitionSpec
    from jaxtyping import Array

    from gemma_4_sql.type_hints import JSONValue
_linear_sig = inspect.signature(nnx.Linear.__init__)
_LINEAR_SUPPORTS_METADATA = "kernel_metadata" in _linear_sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in _linear_sig.parameters.values())
_embed_sig = inspect.signature(nnx.Embed.__init__)
_EMBED_SUPPORTS_METADATA = "embedding_metadata" in _embed_sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in _embed_sig.parameters.values())


def batched_merge_modalities(img_emb: Array, text_emb: Array, token_mask: Array) -> Array:
    """Merge image and text embeddings based on a token mask.

    Args:
    ----
        img_emb: Image embeddings (B, Li, D)
        text_emb: Text embeddings (B, Lt, D)
        token_mask: Boolean mask indicating image token positions (B, Lt)

    Returns:
    -------
        Merged embeddings (B, Lt, D)

    """

    def merge_modalities(i_emb: object, t_emb: object, mask: object) -> object:
        """Merge image and text embeddings using the provided token mask."""
        img_indices = jnp.cumsum(mask) - 1
        safe_indices = jnp.clip(img_indices, 0, i_emb.shape[0] - 1)
        aligned_images = i_emb[safe_indices]
        return jnp.where(mask[:, None], aligned_images, t_emb)

    return jax.vmap(merge_modalities)(img_emb, text_emb, token_mask)


class LayerCache(nnx.Module):
    """KV Cache for a single decoder layer.

    Attributes
    ----------
        k_cache: The key cache tensor.
        v_cache: The value cache tensor.
        cur_ind: The current sequence index being written to.
        size: The maximum sequence length the cache can hold.

    """

    def __init__(self, cache_shape: tuple[int, int, int, int], dtype: jnp.dtype, _shd: PartitionSpec | None = None) -> None:
        """Docstring for __init__."""
        self.k_cache = nnx.Cache(jnp.zeros(cache_shape, dtype=dtype))
        self.v_cache = nnx.Cache(jnp.zeros(cache_shape, dtype=dtype))
        self.cur_ind = nnx.Cache(jnp.zeros((), dtype=jnp.int32))
        self.size = cache_shape[1]


Cache = list[LayerCache]


def init_cache(config: ModelConfig, batch_size: int, max_seq_len: int) -> Cache:
    """Initialize the KV cache for all layers.

    Args:
    ----
        config: The model configuration.
        batch_size: The batch size for generation.
        max_seq_len: The maximum sequence length to cache.

    Returns:
    -------
        A list of LayerCache objects, one for each hidden layer.

    """
    cache_size = 2 ** math.ceil(math.log2(max(max_seq_len, 1)))
    caches = []
    for i in range(config.num_hidden_layers):
        attn_type = GEMMA4_ATTENTION_PATTERN[i % len(GEMMA4_ATTENTION_PATTERN)]
        if attn_type == AttentionType.GLOBAL:
            num_kv = config.num_global_key_value_heads if config.num_global_key_value_heads is not None else config.num_key_value_heads
            hd = config.global_head_dim if config.global_head_dim is not None else config.head_dim
        else:
            num_kv = config.num_key_value_heads
            hd = config.head_dim
        caches.append(LayerCache((batch_size, cache_size, num_kv, hd), config.dtype, config.shd_cfg.cache))
    return caches


GEMMA4_ATTENTION_PATTERN = (AttentionType.LOCAL_SLIDING, AttentionType.LOCAL_SLIDING, AttentionType.LOCAL_SLIDING, AttentionType.LOCAL_SLIDING, AttentionType.LOCAL_SLIDING, AttentionType.GLOBAL)


class Gemma4Model(nnx.Module):
    """The base Gemma 4 trunk consisting of embeddings and a stack of decoder layers."""

    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs) -> None:
        """Docstring for __init__."""
        self.config = config
        shd = config.shd_cfg
        self.embed_tokens = _make_embed(config.vocab_size, config.hidden_size, embedding_metadata={}, rngs=rngs)
        math = __import__("math")
        self.embed_scale = float(math.sqrt(config.hidden_size))
        if config.hidden_size_per_layer_input:
            vocab_size_per_layer = config.vocab_size_per_layer_input if config.vocab_size_per_layer_input is not None else config.vocab_size
            self.embed_tokens_per_layer = _make_embed(vocab_size_per_layer, config.num_hidden_layers * config.hidden_size_per_layer_input, embedding_metadata={}, rngs=rngs)
            self.per_layer_input_scale = 2.0 ** (-0.5)
            self.per_layer_model_projection = _make_linear(config.hidden_size, config.num_hidden_layers * config.hidden_size_per_layer_input, use_bias=False, kernel_metadata={}, bias_metadata={}, rngs=rngs)
            self.per_layer_model_projection_scale = config.hidden_size ** (-0.5)
            self.per_layer_projection_norm = Gemma4RMSNorm(config.hidden_size_per_layer_input, eps=config.rms_norm_eps, dtype=config.dtype, _shd=shd.norm, rngs=rngs)
        self.layers = nnx.List([Gemma4DecoderLayer(config, GEMMA4_ATTENTION_PATTERN[i % len(GEMMA4_ATTENTION_PATTERN)], rngs=rngs) for i in range(config.num_hidden_layers)])
        self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=config.dtype, _shd=shd.norm, rngs=rngs)

    def get_per_layer_inputs(self, input_ids: Array) -> Array:
        """Compute the token-identity component of Per-Layer Embeddings (PLE)."""
        ple = self.embed_tokens_per_layer(input_ids) * self.config.hidden_size_per_layer_input**0.5
        (batch_size, seq_len, _) = ple.shape
        return ple.reshape(batch_size, seq_len, self.config.num_hidden_layers, self.config.hidden_size_per_layer_input)

    def project_per_layer_inputs(self, inputs_embeds: Array, per_layer_inputs: Array | None = None) -> Array:
        """Projects `inputs_embeds` and combines with token-identity `per_layer_inputs`."""
        (batch_size, seq_len, _) = inputs_embeds.shape
        proj = self.per_layer_model_projection(inputs_embeds) * self.per_layer_model_projection_scale
        proj = proj.reshape(batch_size, seq_len, self.config.num_hidden_layers, self.config.hidden_size_per_layer_input)
        proj = self.per_layer_projection_norm(proj)
        if per_layer_inputs is not None:
            proj = (proj + per_layer_inputs) * self.per_layer_input_scale
        return proj

    @jax.named_scope("gemma4_model")
    def __call__(self, input_ids: Array, positions: Array, cache: Cache | None = None, **kwargs: object) -> Array:
        """Apply embeddings and runs the forward pass through all decoder layers.

        Args:
        ----
            input_ids: Token IDs.
            positions: Sequence positions.
            cache: Optional list of KV caches (one per layer).
            **kwargs: Optional args `attention_mask`, `per_layer_inputs`.

        Returns:
        -------
            Hidden states output.

        """
        attention_mask = kwargs.get("attention_mask")
        per_layer_inputs = kwargs.get("per_layer_inputs")
        x = self.embed_tokens(input_ids) * self.embed_scale
        if self.config.hidden_size_per_layer_input:
            if per_layer_inputs is None:
                per_layer_inputs = self.get_per_layer_inputs(input_ids)
            per_layer_inputs = self.project_per_layer_inputs(x, per_layer_inputs)
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            layer_ple = per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
            x = layer(x, positions, layer_cache, attention_mask=attention_mask, per_layer_input=layer_ple)
        return self.norm(x)


def _download_and_load_pretrained(model_name: str, config: ModelConfig | None = None) -> object:
    """Download and load pretrained model."""
    snapshot_download = __import__("huggingface_hub", fromlist=["snapshot_download"]).snapshot_download
    params = __import__(".", fromlist=["params"]).params
    if config is None:
        config_map = {
            "google/gemma-4-E2B": ModelConfig.gemma4_e2b,
            "google/gemma-4-E2B-it": ModelConfig.gemma4_e2b,
            "google/gemma-4-E4B": ModelConfig.gemma4_e4b,
            "google/gemma-4-E4B-it": ModelConfig.gemma4_e4b,
            "google/gemma-4-26B-A4B": ModelConfig.gemma4_26b_a4b,
            "google/gemma-4-26B-A4B-it": ModelConfig.gemma4_26b_a4b,
            "google/gemma-4-31B": ModelConfig.gemma4_31b,
            "google/gemma-4-31B-it": ModelConfig.gemma4_31b,
        }
        if model_name not in config_map:
            msg = f"Model name '{model_name}' is unknown, please provide config argument"
            raise ValueError(msg)
        config = config_map[model_name]()
    model_ckpt_path = snapshot_download(repo_id=model_name, allow_patterns="*.safetensors")
    return params.create_gemma4_from_pretrained(model_ckpt_path, config)


class Gemma4ForCausalLM(nnx.Module):
    """Gemma 4 model with a language modeling head."""

    @classmethod
    def from_pretrained(cls, model_name: str, config: ModelConfig | None = None) -> object:
        """Load from pretrained."""
        return _download_and_load_pretrained(model_name, config)

    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs) -> None:
        """Docstring for __init__."""
        self.config = config
        self.model = Gemma4Model(config, rngs=rngs)
        self.lm_head = _make_linear(config.hidden_size, config.vocab_size, use_bias=False, kernel_metadata={}, bias_metadata={}, rngs=rngs)
        self.vision_tower = SiglipVisionTransformer(config.vision_config, rngs=rngs) if config.vision_config else None
        self.multi_modal_projector = Gemma4MultiModalProjector(config, config.vision_config, config.mm_tokens_per_image, rngs=rngs) if config.vision_config else None
        self.audio_tower = Gemma4AudioModel(config.audio_config, rngs=rngs) if config.audio_config else None
        if config.audio_config:
            multimodal_hidden_size = getattr(config.audio_config, "output_proj_dims", config.audio_config.hidden_size)
            self.embed_audio = Gemma4MultimodalEmbedder(multimodal_hidden_size, config.hidden_size, config.audio_config.rms_norm_eps, rngs=rngs)
        else:
            self.embed_audio = None

    def _merge_multimodal_features(self, inputs_embeds: Array, image_features: Array | None, image_token_mask: Array | None, audio_features: Array | None, audio_token_mask: Array | None) -> Array:
        """Merge vision and audio features into the text embeddings."""
        if image_features is not None and image_token_mask is not None:
            inputs_embeds = batched_merge_modalities(image_features, inputs_embeds, image_token_mask)
        if audio_features is not None and audio_token_mask is not None:
            inputs_embeds = batched_merge_modalities(audio_features, inputs_embeds, audio_token_mask)
        return inputs_embeds

    def _process_multimodal(self, pixel_values: Array | None, image_token_mask: Array | None, input_features: Array | None, input_features_mask: Array | None, audio_token_mask: Array | None, input_ids: Array) -> tuple[Array | None, bool]:
        """Process multimodal inputs and embed them."""
        has_vision = pixel_values is not None and self.vision_tower is not None
        has_audio = input_features is not None and self.audio_tower is not None
        if not has_vision and (not has_audio):
            return (None, False)
        inputs_embeds = self.model.embed_tokens(input_ids) * self.model.embed_scale
        image_features = None
        audio_features = None
        if has_vision:
            vision_outputs = self.vision_tower(pixel_values)
            image_features = self.multi_modal_projector(vision_outputs)
        if has_audio:
            audio_outputs = self.audio_tower(input_features, input_features_mask)
            audio_features = self.embed_audio(audio_outputs)
        inputs_embeds = self._merge_multimodal_features(inputs_embeds, image_features, image_token_mask, audio_features, audio_token_mask)
        return (inputs_embeds, True)

    def _apply_layers_multimodal(self, hidden_states: Array, input_ids: Array, positions: Array, cache: Cache | None, attention_mask: Array | None) -> Array:
        """Apply model layers on multimodal hidden states."""
        if self.config.hidden_size_per_layer_input:
            per_layer_inputs_id = self.model.get_per_layer_inputs(input_ids)
            per_layer_inputs = self.model.project_per_layer_inputs(hidden_states, per_layer_inputs_id)
        else:
            per_layer_inputs = None
        for i, layer in enumerate(self.model.layers):
            layer_cache = cache[i] if cache is not None else None
            layer_ple = per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
            hidden_states = layer(hidden_states, positions, layer_cache, attention_mask=attention_mask, per_layer_input=layer_ple)
        return self.model.norm(hidden_states)

    @jax.named_scope("gemma4_causal_lm")
    def __call__(self, input_ids: Array, positions: Array, cache: Cache | None = None, **kwargs: object) -> Array:
        """Compute logits for the given sequence, optionally applying soft-capping."""
        attention_mask = kwargs.get("attention_mask")
        (inputs_embeds, is_multimodal) = self._process_multimodal(kwargs.get("pixel_values"), kwargs.get("image_token_mask"), kwargs.get("input_features"), kwargs.get("input_features_mask"), kwargs.get("audio_token_mask"), input_ids)
        hidden_states = self._apply_layers_multimodal(inputs_embeds, input_ids, positions, cache, attention_mask) if is_multimodal and inputs_embeds is not None else self.model(input_ids, positions, cache, attention_mask=attention_mask)
        logits = self.lm_head(hidden_states)
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = jnp.tanh(logits) * self.config.final_logit_softcapping
        return logits.astype(jnp.float32)


@nnx.jit
def forward(model: nnx.Module, cache: Cache, input_ids: Array, positions: Array, pixel_values: Array | None = None, **kwargs: JSONValue) -> tuple[Array, Cache]:
    """Execute a standard forward pass returning logits and updated cache."""
    image_token_mask = kwargs.get("image_token_mask")
    input_features = kwargs.get("input_features")
    input_features_mask = kwargs.get("input_features_mask")
    audio_token_mask = kwargs.get("audio_token_mask")
    logits = model(input_ids=input_ids, positions=positions, cache=cache, pixel_values=pixel_values, image_token_mask=image_token_mask, input_features=input_features, input_features_mask=input_features_mask, audio_token_mask=audio_token_mask)
    return (logits[:, -1, :], cache)

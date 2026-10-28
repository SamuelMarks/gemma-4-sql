"""Gemma 4 model implementation in JAX/Flax NNX."""

from __future__ import annotations

import math
from collections.abc import Callable

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
    compute_audio_attention_outputs,
    convert_to_block,
    extract_block_context,
    rel_shift,
)
from .cache import GEMMA4_ATTENTION_PATTERN, Cache
from .config import AttentionType, AudioConfig, ModelConfig, ModelConfigPresets, ShardConfig, ShardMode, VisionConfig, VisionShardConfig
from .decoder_layer import Gemma4DecoderLayer
from .layers import ConstVar, Gemma4ClippableLinear, Gemma4MLP, Gemma4RMSNorm, LinearLayerProtocol, StatVar, make_embed, make_linear
from .moe import Gemma4MoE, Gemma4RoutedExperts
from .multimodal import MultimodalInputs, batched_merge_modalities
from .params import create_gemma4_from_pretrained
from .vision import Gemma4MultimodalEmbedder, Gemma4MultiModalProjector, SiglipAttention, SiglipEncoderLayer, SiglipMLP, SiglipVisionEmbeddings, SiglipVisionTransformer, avg_pool_vision_outputs

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
    "LinearLayerProtocol",
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
    "_default_jit",
    "avg_pool_vision_outputs",
    "compute_audio_attention_outputs",
    "convert_to_block",
    "extract_block_context",
    "forward",
    "make_embed",
    "make_linear",
    "rel_shift",
]


class Gemma4Model(nnx.Module):
    """The base Gemma 4 trunk consisting of embeddings and a stack of decoder layers."""

    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs) -> None:
        """Initialize Gemma4Model trunk.

        Args:
            config: Architectural configuration parameters.
            rngs: Flax NNX random number generator streams.

        """
        self.config = config
        shd = config.shd_cfg
        self.embed_tokens = make_embed(config.vocab_size, config.hidden_size, embedding_metadata={}, rngs=rngs)
        self.embed_scale = float(math.sqrt(config.hidden_size))
        if config.hidden_size_per_layer_input:
            vocab_size_per_layer = config.vocab_size_per_layer_input if config.vocab_size_per_layer_input is not None else config.vocab_size
            self.embed_tokens_per_layer = make_embed(vocab_size_per_layer, config.num_hidden_layers * config.hidden_size_per_layer_input, embedding_metadata={}, rngs=rngs)
            self.per_layer_input_scale = 2.0 ** (-0.5)
            self.per_layer_model_projection = make_linear(config.hidden_size, config.num_hidden_layers * config.hidden_size_per_layer_input, use_bias=False, kernel_metadata={}, bias_metadata={}, rngs=rngs)
            self.per_layer_model_projection_scale = config.hidden_size ** (-0.5)
            self.per_layer_projection_norm = Gemma4RMSNorm(config.hidden_size_per_layer_input, eps=config.rms_norm_eps, dtype=config.dtype, _shd=shd.norm, rngs=rngs)
        self.layers = nnx.List([Gemma4DecoderLayer(config, GEMMA4_ATTENTION_PATTERN[i % len(GEMMA4_ATTENTION_PATTERN)], rngs=rngs) for i in range(config.num_hidden_layers)])
        self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=config.dtype, _shd=shd.norm, rngs=rngs)

    def inject_lora_projections(
        self,
        target_modules: list[str],
        adapter_factory: Callable[[nnx.Linear], LinearLayerProtocol],
    ) -> int:
        """Wrap matching attention and MLP projections across all decoder layers with LoRA adapters.

        Args:
            target_modules: List of target projection names to wrap.
            adapter_factory: Factory callable converting an `nnx.Linear` into a `LinearLayerProtocol` adapter.

        Returns:
            Total number of projections wrapped across all decoder layers.

        """
        injected = 0
        for layer in self.layers:
            injected += layer.inject_lora_projections(target_modules, adapter_factory)
        return injected

    def get_per_layer_inputs(self, input_ids: Array) -> Array:
        """Compute the token-identity component of Per-Layer Embeddings (PLE).

        Args:
            input_ids: Input token ID array of shape `(batch_size, seq_len)`.

        Returns:
            Per-layer input embedding array.

        """
        if self.config.hidden_size_per_layer_input is None:
            return input_ids
        ple = self.embed_tokens_per_layer(input_ids) * float(self.config.hidden_size_per_layer_input) ** 0.5
        (batch_size, seq_len, _) = ple.shape
        res: Array = ple.reshape(batch_size, seq_len, self.config.num_hidden_layers, self.config.hidden_size_per_layer_input)
        return res

    def project_per_layer_inputs(self, inputs_embeds: Array, per_layer_inputs: Array | None = None) -> Array:
        """Project `inputs_embeds` and combine with token-identity `per_layer_inputs`.

        Args:
            inputs_embeds: Input token embeddings of shape `(batch_size, seq_len, hidden_size)`.
            per_layer_inputs: Optional precomputed token-identity per-layer embeddings.

        Returns:
            Projected and normalized per-layer embedding tensor.

        """
        (batch_size, seq_len, _) = inputs_embeds.shape
        ple_dim = self.config.hidden_size_per_layer_input or 0
        proj = self.per_layer_model_projection(inputs_embeds) * self.per_layer_model_projection_scale
        proj = proj.reshape(batch_size, seq_len, self.config.num_hidden_layers, ple_dim)
        proj = self.per_layer_projection_norm(proj)
        if per_layer_inputs is not None:
            proj = (proj + per_layer_inputs) * self.per_layer_input_scale
        return proj

    @jax.named_scope("gemma4_model")
    def __call__(
        self,
        input_ids: Array,
        positions: Array,
        cache: Cache | None = None,
        *,
        attention_mask: Array | None = None,
        per_layer_inputs: Array | None = None,
    ) -> Array:
        """Apply embeddings and run the forward pass through all decoder layers.

        Args:
            input_ids: Token IDs tensor.
            positions: Sequence positions tensor.
            cache: Optional list of KV caches (one per layer).
            attention_mask: Optional additive attention mask.
            per_layer_inputs: Optional precomputed per-layer input embeddings.

        Returns:
            Normalized final hidden states tensor.

        """
        x = self.embed_tokens(input_ids) * self.embed_scale
        if self.config.hidden_size_per_layer_input:
            if per_layer_inputs is None:
                per_layer_inputs = self.get_per_layer_inputs(input_ids)
            per_layer_inputs = self.project_per_layer_inputs(x, per_layer_inputs)
        num_layers = len(self.layers)
        kv_shared = int(self.config.num_kv_shared_layers or 0)
        pivot = max(1, num_layers - kv_shared) if kv_shared > 0 else num_layers
        for i, layer in enumerate(self.layers):
            cache_idx = (i % pivot) if (kv_shared > 0 and i >= pivot) else i
            layer_cache = cache[cache_idx] if cache is not None else None
            layer_ple = per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
            x = layer(x, positions, layer_cache, attention_mask=attention_mask, per_layer_input=layer_ple)
        return self.norm(x)


def _download_and_load_pretrained(model_name: str, config: ModelConfig | None = None) -> Gemma4ForCausalLM:
    """Download and load a pretrained Gemma 4 model from Hugging Face Hub.

    Args:
        model_name: Hugging Face repository identifier.
        config: Optional explicit `ModelConfig` override.

    Returns:
        Instantiated `Gemma4ForCausalLM` model with loaded weights.

    Raises:
        ValueError: If `model_name` is unknown and `config` is not provided.

    """
    from huggingface_hub import snapshot_download

    if config is None:
        config_map: dict[str, Callable[[], ModelConfig]] = {
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
        resolved_config = config_map[model_name]()
    else:
        resolved_config = config
    model_ckpt_path = snapshot_download(repo_id=model_name, allow_patterns="*.safetensors")
    return create_gemma4_from_pretrained(model_ckpt_path, resolved_config)


class Gemma4ForCausalLM(nnx.Module):
    """Gemma 4 model with a language modeling head and optional vision/audio towers."""

    vision_tower: SiglipVisionTransformer | None
    multi_modal_projector: Gemma4MultiModalProjector | None
    audio_tower: Gemma4AudioModel | None
    embed_audio: Gemma4MultimodalEmbedder | None

    @classmethod
    def from_pretrained(cls, model_name: str, config: ModelConfig | None = None) -> Gemma4ForCausalLM:
        """Load a `Gemma4ForCausalLM` model from pretrained weights.

        Args:
            model_name: Hugging Face model identifier or local path.
            config: Optional `ModelConfig` override.

        Returns:
            Loaded `Gemma4ForCausalLM` instance.

        """
        return _download_and_load_pretrained(model_name, config)

    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs) -> None:
        """Initialize `Gemma4ForCausalLM`.

        Args:
            config: Model configuration.
            rngs: Flax NNX random number generator streams.

        """
        self.config = config
        self.model = Gemma4Model(config, rngs=rngs)
        self.lm_head = make_linear(config.hidden_size, config.vocab_size, use_bias=False, kernel_metadata={}, bias_metadata={}, rngs=rngs)
        self.vision_tower = SiglipVisionTransformer(config.vision_config, rngs=rngs) if config.vision_config else None
        self.multi_modal_projector = Gemma4MultiModalProjector(config, rngs=rngs) if config.vision_config else None
        self.audio_tower = Gemma4AudioModel(config.audio_config, rngs=rngs) if config.audio_config else None
        if config.audio_config:
            self.embed_audio = Gemma4MultimodalEmbedder(config, rngs=rngs)
        else:
            self.embed_audio = None

    def inject_lora_projections(
        self,
        target_modules: list[str],
        adapter_factory: Callable[[nnx.Linear], LinearLayerProtocol],
    ) -> int:
        """Delegate LoRA projection wrapping to the inner `Gemma4Model` trunk.

        Args:
            target_modules: List of target projection names to wrap.
            adapter_factory: Factory callable converting an `nnx.Linear` into a `LinearLayerProtocol` adapter.

        Returns:
            Total number of projections wrapped across the model.

        """
        return self.model.inject_lora_projections(target_modules, adapter_factory)

    @staticmethod
    def _merge_multimodal_features(
        inputs_embeds: Array,
        image_features: Array | None,
        audio_features: Array | None,
        inputs: MultimodalInputs,
    ) -> Array:
        """Merge vision and audio features into the text token embeddings.

        Args:
            inputs_embeds: Base text embeddings.
            image_features: Optional projected vision embeddings.
            audio_features: Optional projected audio embeddings.
            inputs: Multimodal input container with token masks.

        Returns:
            Merged multimodal embedding tensor.

        """
        if image_features is not None and inputs.image_token_mask is not None:
            inputs_embeds = batched_merge_modalities(image_features, inputs_embeds, inputs.image_token_mask)
        if audio_features is not None and inputs.audio_token_mask is not None:
            inputs_embeds = batched_merge_modalities(audio_features, inputs_embeds, inputs.audio_token_mask)
        return inputs_embeds

    def _process_multimodal(self, inputs: MultimodalInputs) -> tuple[Array | None, bool]:
        """Process multimodal inputs and embed them into the text embedding space.

        Args:
            inputs: Multimodal inputs bundle.

        Returns:
            Tuple of `(merged_embeddings_or_none, is_multimodal)`.

        """
        has_vision = inputs.pixel_values is not None and self.vision_tower is not None
        has_audio = inputs.input_features is not None and self.audio_tower is not None
        if not has_vision and (not has_audio):
            return (None, False)
        inputs_embeds = self.model.embed_tokens(inputs.input_ids) * self.model.embed_scale
        image_features = None
        audio_features = None
        if has_vision and self.vision_tower is not None and self.multi_modal_projector is not None and inputs.pixel_values is not None:
            vision_outputs = self.vision_tower(inputs.pixel_values)
            image_features = self.multi_modal_projector(vision_outputs)
        if has_audio and self.audio_tower is not None and self.embed_audio is not None and inputs.input_features is not None:
            audio_outputs = self.audio_tower(inputs.input_features, inputs.input_features_mask)
            audio_features = self.embed_audio(audio_outputs)
        inputs_embeds = self._merge_multimodal_features(inputs_embeds, image_features, audio_features, inputs)
        return (inputs_embeds, True)

    def _apply_layers_multimodal(
        self,
        hidden_states: Array,
        inputs: MultimodalInputs,
        positions: Array,
        cache: Cache | None,
    ) -> Array:
        """Apply decoder layers on merged multimodal hidden states.

        Args:
            hidden_states: Merged multimodal embeddings.
            inputs: Multimodal inputs bundle.
            positions: Sequence position indices.
            cache: Optional KV cache.

        Returns:
            Normalized decoder output hidden states.

        """
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        if self.config.hidden_size_per_layer_input:
            per_layer_inputs_id = self.model.get_per_layer_inputs(input_ids)
            per_layer_inputs = self.model.project_per_layer_inputs(hidden_states, per_layer_inputs_id)
        else:
            per_layer_inputs = None
        num_layers = len(self.model.layers)
        kv_shared = int(self.config.num_kv_shared_layers or 0)
        pivot = max(1, num_layers - kv_shared) if kv_shared > 0 else num_layers
        for i, layer in enumerate(self.model.layers):
            cache_idx = (i % pivot) if (kv_shared > 0 and i >= pivot) else i
            layer_cache = cache[cache_idx] if cache is not None else None
            layer_ple = per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
            hidden_states = layer(hidden_states, positions, layer_cache, attention_mask=attention_mask, per_layer_input=layer_ple)
        return self.model.norm(hidden_states)

    @jax.named_scope("gemma4_causal_lm")
    def __call__(
        self,
        input_ids: Array,
        positions: Array | None = None,
        cache: Cache | None = None,
        *,
        attention_mask: Array | None = None,
        pixel_values: Array | None = None,
        image_token_mask: Array | None = None,
        input_features: Array | None = None,
        input_features_mask: Array | None = None,
        audio_token_mask: Array | None = None,
    ) -> Array:
        """Compute logits for the given sequence, optionally applying soft-capping.

        Args:
            input_ids: Input sequence token IDs.
            positions: Optional sequence position indices (defaults to `0..seq_len-1`).
            cache: Key-value cache for autoregressive decoding.
            attention_mask: Optional additive attention mask.
            pixel_values: Optional image pixel values tensor.
            image_token_mask: Optional boolean mask for image tokens.
            input_features: Optional audio spectrogram features tensor.
            input_features_mask: Optional audio frame mask tensor.
            audio_token_mask: Optional boolean mask for audio tokens.

        Returns:
            Logits tensor for the output vocabulary.

        """
        if positions is None:
            positions = jnp.arange(input_ids.shape[-1])[None, :]
        mm_inputs = MultimodalInputs(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_token_mask=image_token_mask,
            input_features=input_features,
            input_features_mask=input_features_mask,
            audio_token_mask=audio_token_mask,
            attention_mask=attention_mask,
        )
        (inputs_embeds, is_multimodal) = self._process_multimodal(mm_inputs)
        hidden_states = self._apply_layers_multimodal(inputs_embeds, mm_inputs, positions, cache) if is_multimodal and inputs_embeds is not None else self.model(input_ids, positions, cache, attention_mask=attention_mask)
        logits = self.lm_head(hidden_states)
        if self.config.final_logit_softcapping is not None:
            logits /= self.config.final_logit_softcapping
            logits = jnp.tanh(logits) * self.config.final_logit_softcapping
        res_logits: Array = logits.astype(jnp.float32)
        return res_logits


def _default_jit[FnT](x: FnT) -> FnT:
    """Provide identity fallback for jit decorator if `nnx.jit` is unavailable.

    Args:
        x: The function to be jitted.

    Returns:
        The original function unmodified.

    """
    return x


@nnx.jit
def forward(
    model: Gemma4ForCausalLM,
    cache: Cache,
    input_ids: Array,
    positions: Array,
    *,
    pixel_values: Array | None = None,
    image_token_mask: Array | None = None,
    input_features: Array | None = None,
    input_features_mask: Array | None = None,
    audio_token_mask: Array | None = None,
) -> tuple[Array, Cache]:
    """Execute a standard forward pass returning next-token logits and updated cache.

    Args:
        model: Gemma4 causal language model instance.
        cache: Key-value attention cache.
        input_ids: Input sequence token IDs.
        positions: Sequence position indices.
        pixel_values: Optional image pixel values tensor.
        image_token_mask: Optional boolean mask for image tokens.
        input_features: Optional audio spectrogram features tensor.
        input_features_mask: Optional audio frame mask tensor.
        audio_token_mask: Optional boolean mask for audio tokens.

    Returns:
        A tuple of `(next_token_logits, updated_cache)`.

    """
    logits = model(
        input_ids=input_ids,
        positions=positions,
        cache=cache,
        pixel_values=pixel_values,
        image_token_mask=image_token_mask,
        input_features=input_features,
        input_features_mask=input_features_mask,
        audio_token_mask=audio_token_mask,
    )
    return (logits[:, -1, :], cache)

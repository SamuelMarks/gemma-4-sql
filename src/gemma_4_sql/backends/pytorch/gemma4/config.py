"""Configuration classes and architectural presets for the Gemma 4 PyTorch implementation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from gemma_4_sql.backends.common_gemma4_config import (
    DEFAULT_GRADIENT_CLIPPING,
    GEMMA4_26B_A4B_SPEC,
    GEMMA4_31B_SPEC,
    GEMMA4_ATTENTION_PATTERN,
    GEMMA4_E2B_SPEC,
    GEMMA4_E4B_SPEC,
    GEMMA4_TINY_SPEC,
    AttentionType,
    BaseAudioConfigFields,
    BaseGemma4ConfigFields,
    BaseVisionConfigFields,
    resolve_gemma4_preset_name,
)
from gemma_4_sql.type_hints import (
    JSONValue,
    extract_bool_kwarg,
    extract_float_kwarg,
    extract_int_kwarg,
)


@dataclass
class Gemma4VisionConfig(BaseVisionConfigFields):
    """Configuration for Gemma 4 Vision sub-model."""

    image_size: int = 224
    pooling_kernel_size: int = 1
    extra_attributes: dict[str, JSONValue] = field(default_factory=dict)

    def __init__(
        self,
        hidden_size: int = 1152,
        intermediate_size: int = 4304,
        num_hidden_layers: int = 27,
        num_attention_heads: int = 16,
        patch_size: int = 14,
        image_size: int = 224,
        layer_norm_eps: float = 1e-06,
        num_channels: int = 3,
        pooling_kernel_size: int = 1,
        **kwargs: JSONValue,
    ) -> None:
        """Initialize the Gemma4VisionConfig.

        Args:
            hidden_size: Vision hidden embedding dimension.
            intermediate_size: Vision MLP intermediate dimension.
            num_hidden_layers: Number of transformer encoder layers.
            num_attention_heads: Number of vision attention heads.
            patch_size: Spatial patch size in pixels.
            image_size: Input image height/width in pixels.
            layer_norm_eps: Layer normalization epsilon constant.
            num_channels: Number of input image channels.
            pooling_kernel_size: Spatial 2D average pooling kernel size.
            **kwargs: Additional configuration attributes.

        """
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.patch_size = patch_size
        self.image_size = image_size
        self.layer_norm_eps = layer_norm_eps
        self.num_channels = num_channels
        self.pooling_kernel_size = pooling_kernel_size
        self.extra_attributes = dict(kwargs)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, JSONValue]) -> Gemma4VisionConfig:
        """Construct a `Gemma4VisionConfig` from a dictionary of configuration values.

        Args:
            mapping: Dictionary of vision configuration keys and values.

        Returns:
            Constructed `Gemma4VisionConfig` instance.

        """
        known_keys = {
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "patch_size",
            "image_size",
            "layer_norm_eps",
            "num_channels",
            "pooling_kernel_size",
        }
        extra = {k: v for k, v in mapping.items() if k not in known_keys}
        return cls(
            hidden_size=extract_int_kwarg(mapping, "hidden_size", 1152),
            intermediate_size=extract_int_kwarg(mapping, "intermediate_size", 4304),
            num_hidden_layers=extract_int_kwarg(mapping, "num_hidden_layers", 27),
            num_attention_heads=extract_int_kwarg(mapping, "num_attention_heads", 16),
            patch_size=extract_int_kwarg(mapping, "patch_size", 14),
            image_size=extract_int_kwarg(mapping, "image_size", 224),
            layer_norm_eps=extract_float_kwarg(mapping, "layer_norm_eps", 1e-06),
            num_channels=extract_int_kwarg(mapping, "num_channels", 3),
            pooling_kernel_size=extract_int_kwarg(mapping, "pooling_kernel_size", 1),
            **extra,
        )


@dataclass
class Gemma4AudioConfig(BaseAudioConfigFields):
    """Configuration for Gemma 4 Audio sub-model."""

    num_mel_bins: int = 80
    extra_attributes: dict[str, JSONValue] = field(default_factory=dict)

    def __init__(
        self,
        hidden_size: int = 1024,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 8,
        rms_norm_eps: float = 1e-06,
        conv_kernel_size: int = 5,
        residual_weight: float = 0.5,
        num_mel_bins: int = 80,
        use_clipped_linears: bool = True,
        gradient_clipping: float = DEFAULT_GRADIENT_CLIPPING,
        output_proj_dims: int = 1536,
        **kwargs: JSONValue,
    ) -> None:
        """Initialize the Gemma4AudioConfig.

        Args:
            hidden_size: Audio encoder hidden dimension.
            num_hidden_layers: Number of audio conformer layers.
            num_attention_heads: Number of audio attention heads.
            rms_norm_eps: RMSNorm epsilon constant.
            conv_kernel_size: Conformer LightConv1d kernel size.
            residual_weight: Macaron feed-forward residual scaling factor.
            num_mel_bins: Number of input mel-spectrogram frequency bins when 3D features are passed.
            use_clipped_linears: Whether linear projections apply gradient clipping.
            gradient_clipping: Maximum absolute activation/gradient clip threshold.
            output_proj_dims: Output projection dimension.
            **kwargs: Additional configuration attributes.

        """
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.rms_norm_eps = rms_norm_eps
        self.conv_kernel_size = conv_kernel_size
        self.residual_weight = residual_weight
        self.num_mel_bins = num_mel_bins
        self.use_clipped_linears = use_clipped_linears
        self.gradient_clipping = gradient_clipping
        self.output_proj_dims = output_proj_dims
        self.extra_attributes = dict(kwargs)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, JSONValue]) -> Gemma4AudioConfig:
        """Construct a `Gemma4AudioConfig` from a dictionary of configuration values.

        Args:
            mapping: Dictionary of audio configuration keys and values.

        Returns:
            Constructed `Gemma4AudioConfig` instance.

        """
        known_keys = {
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "rms_norm_eps",
            "conv_kernel_size",
            "residual_weight",
            "num_mel_bins",
            "use_clipped_linears",
            "gradient_clipping",
            "output_proj_dims",
        }
        extra = {k: v for k, v in mapping.items() if k not in known_keys}
        return cls(
            hidden_size=extract_int_kwarg(mapping, "hidden_size", 1024),
            num_hidden_layers=extract_int_kwarg(mapping, "num_hidden_layers", 12),
            num_attention_heads=extract_int_kwarg(mapping, "num_attention_heads", 8),
            rms_norm_eps=extract_float_kwarg(mapping, "rms_norm_eps", 1e-06),
            conv_kernel_size=extract_int_kwarg(mapping, "conv_kernel_size", 5),
            residual_weight=extract_float_kwarg(mapping, "residual_weight", 0.5),
            num_mel_bins=extract_int_kwarg(mapping, "num_mel_bins", 80),
            use_clipped_linears=extract_bool_kwarg(mapping, "use_clipped_linears", True),
            gradient_clipping=extract_float_kwarg(mapping, "gradient_clipping", DEFAULT_GRADIENT_CLIPPING),
            output_proj_dims=extract_int_kwarg(mapping, "output_proj_dims", 1536),
            **extra,
        )


@dataclass
class Gemma4Config(BaseGemma4ConfigFields):
    """Configuration for native PyTorch Gemma 4."""

    pad_token_id: int = 0
    eos_token_id: int | None = None
    image_token_id: int = 255999
    audio_token_id: int | None = 255998
    router_jitter_noise: float = 0.0
    sliding_window: int | None = 4096
    global_attn_layers: Sequence[int] = field(default_factory=list)
    rope_theta: float = 10000.0
    partial_rotary_factor: float = 1.0
    max_position_embeddings: int = 8192
    is_global: bool = False
    audio_hidden_size: int = 1024
    vision_config: Gemma4VisionConfig = field(default_factory=Gemma4VisionConfig)
    audio_config: Gemma4AudioConfig = field(default_factory=Gemma4AudioConfig)
    extra_attributes: dict[str, JSONValue] = field(default_factory=dict)

    def __init__(
        self,
        vocab_size: int = 256000,
        hidden_size: int = 2048,
        num_hidden_layers: int = 18,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 1,
        intermediate_size: int = 16384,
        rms_norm_eps: float = 1e-6,
        head_dim: int = 256,
        pad_token_id: int = 0,
        image_token_id: int = 256000,
        audio_token_id: int | None = 256001,
        num_experts: int = 4,
        num_experts_per_tok: int = 2,
        router_jitter_noise: float = 0.0,
        sliding_window: int | None = 4096,
        global_attn_layers: Sequence[int] | None = None,
        rope_theta: float = 10000.0,
        partial_rotary_factor: float = 1.0,
        vision_config: Mapping[str, JSONValue] | Gemma4VisionConfig | None = None,
        audio_config: Mapping[str, JSONValue] | Gemma4AudioConfig | None = None,
        moe_intermediate_size: int | None = None,
        global_head_dim: int | None = None,
        eos_token_id: int | None = None,
        num_global_key_value_heads: int | None = None,
        share_kv_projections: bool = False,
        num_shared_experts: int = 1,
        num_kv_shared_layers: int = 0,
        use_post_norms: bool = False,
        vocab_size_per_layer_input: int | None = None,
        hidden_size_per_layer_input: int | None = None,
        rope_max_timescale: int = 10000,
        global_rope_max_timescale: int | None = 1000000,
        local_rope_max_timescale: int | None = None,
        local_rope_proportion: float = 1.0,
        global_rope_proportion: float = 0.25,
        float32_gate_logits: bool = True,
        final_logit_softcapping: float | None = None,
        attn_logits_soft_cap: float | None = None,
        max_position_embeddings: int = 8192,
        is_global: bool = False,
        audio_hidden_size: int | None = None,
        **kwargs: JSONValue,
    ) -> None:
        """Initialize the Gemma4Config.

        Args:
            vocab_size: Vocabulary size.
            hidden_size: Model hidden dimension.
            num_hidden_layers: Number of decoder layers.
            num_attention_heads: Number of query attention heads.
            num_key_value_heads: Number of key-value heads (GQA/MQA).
            intermediate_size: Dense MLP intermediate size.
            rms_norm_eps: RMSNorm epsilon constant.
            head_dim: Attention head dimension.
            pad_token_id: Padding token ID.
            image_token_id: Image placeholder token ID.
            audio_token_id: Audio placeholder token ID.
            num_experts: Total number of routed MoE experts.
            num_experts_per_tok: Number of active experts per token.
            router_jitter_noise: Router jitter noise during training.
            sliding_window: Sliding window attention span.
            global_attn_layers: Indices of global attention layers.
            rope_theta: Rotary positional embedding base frequency.
            partial_rotary_factor: Fraction of head dimension receiving RoPE.
            vision_config: Optional vision sub-configuration dictionary or instance.
            audio_config: Optional audio sub-configuration dictionary or instance.
            moe_intermediate_size: Optional MoE expert intermediate dimension.
            global_head_dim: Optional global attention head dimension.
            eos_token_id: Optional end-of-sequence token ID.
            num_global_key_value_heads: Optional KV head count for global attention layers.
            share_kv_projections: Whether global attention layers share K and V projections.
            num_shared_experts: Number of always-active shared experts in MoE blocks.
            num_kv_shared_layers: Number of top decoder layers sharing KV states from earlier layers.
            use_post_norms: Whether to apply post-attention and post-FFN RMSNorms.
            vocab_size_per_layer_input: Optional vocabulary size for Per-Layer Embeddings (PLE).
            hidden_size_per_layer_input: Optional hidden dimension for Per-Layer Embeddings (PLE).
            rope_max_timescale: Default RoPE max timescale.
            global_rope_max_timescale: Global attention RoPE max timescale.
            local_rope_max_timescale: Optional local sliding attention RoPE max timescale.
            local_rope_proportion: Fraction of head dimension rotated in local layers.
            global_rope_proportion: Fraction of head dimension rotated in global layers.
            float32_gate_logits: Whether router gate logits are computed in float32.
            final_logit_softcapping: Optional soft-cap value for final vocabulary logits.
            attn_logits_soft_cap: Optional soft-cap value for attention score logits.
            max_position_embeddings: Maximum sequence length for rotary embeddings.
            is_global: Explicit flag indicating whether attention is global.
            audio_hidden_size: Optional override for audio encoder hidden size.
            **kwargs: Additional configuration attributes.

        """
        self.vocab_size = vocab_size
        self.vocab_size_per_layer_input = vocab_size_per_layer_input
        self.hidden_size = hidden_size
        self.hidden_size_per_layer_input = hidden_size_per_layer_input or None
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_global_key_value_heads = num_global_key_value_heads
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.rms_norm_eps = rms_norm_eps
        self.head_dim = head_dim
        self.global_head_dim = global_head_dim
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.image_token_id = image_token_id
        self.audio_token_id = audio_token_id

        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_shared_experts = num_shared_experts
        self.num_kv_shared_layers = num_kv_shared_layers
        self.use_post_norms = use_post_norms
        self.share_kv_projections = share_kv_projections
        self.router_jitter_noise = router_jitter_noise

        self.sliding_window = sliding_window
        self.sliding_window_size = sliding_window if sliding_window is not None else 512
        if global_attn_layers is not None:
            self.global_attn_layers = list(global_attn_layers)
        else:
            self.global_attn_layers = [idx for idx in range(num_hidden_layers) if GEMMA4_ATTENTION_PATTERN[idx % len(GEMMA4_ATTENTION_PATTERN)] == AttentionType.GLOBAL]

        self.rope_theta = rope_theta
        self.rope_max_timescale = rope_max_timescale
        self.global_rope_max_timescale = global_rope_max_timescale
        self.local_rope_max_timescale = local_rope_max_timescale
        self.local_rope_proportion = local_rope_proportion
        self.global_rope_proportion = global_rope_proportion
        self.partial_rotary_factor = partial_rotary_factor
        self.float32_gate_logits = float32_gate_logits
        self.final_logit_softcapping = final_logit_softcapping
        self.attn_logits_soft_cap = attn_logits_soft_cap
        self.max_position_embeddings = max_position_embeddings
        self.is_global = is_global
        raw_mm_tokens = kwargs.pop("mm_tokens_per_image", 256)
        self.mm_tokens_per_image = int(raw_mm_tokens) if isinstance(raw_mm_tokens, (int, float, str)) else 256

        if vision_config is None:
            if hidden_size < 1024:
                self.vision_config = Gemma4VisionConfig(
                    hidden_size=min(64, max(16, hidden_size)),
                    intermediate_size=min(128, max(32, intermediate_size)),
                    num_hidden_layers=min(2, max(1, num_hidden_layers)),
                    num_attention_heads=min(4, max(1, num_attention_heads)),
                    image_size=224,
                    patch_size=14,
                )
            else:
                self.vision_config = Gemma4VisionConfig()
        elif isinstance(vision_config, Gemma4VisionConfig):
            self.vision_config = vision_config
        else:
            self.vision_config = Gemma4VisionConfig.from_mapping(vision_config)

        if audio_config is None:
            if hidden_size < 1024:
                self.audio_config = Gemma4AudioConfig(
                    hidden_size=min(64, max(16, hidden_size)),
                    num_hidden_layers=min(2, max(1, num_hidden_layers)),
                    num_attention_heads=min(4, max(1, num_attention_heads)),
                )
            else:
                self.audio_config = Gemma4AudioConfig()
        elif isinstance(audio_config, Gemma4AudioConfig):
            self.audio_config = audio_config
        else:
            self.audio_config = Gemma4AudioConfig.from_mapping(audio_config)

        self.audio_hidden_size = audio_hidden_size if audio_hidden_size is not None else self.audio_config.hidden_size
        self.extra_attributes = dict(kwargs)

    @classmethod
    def _from_spec(cls, spec: Mapping[str, int]) -> Gemma4Config:
        """Instantiate a `Gemma4Config` from a preset specification mapping.

        Args:
            spec: Mapping of integer architectural hyperparameters.

        Returns:
            Configured `Gemma4Config` instance.

        """
        return cls(
            vocab_size=spec.get("vocab_size", 262144),
            hidden_size=spec.get("hidden_size", 2048),
            intermediate_size=spec.get("intermediate_size", 8192),
            moe_intermediate_size=spec.get("moe_intermediate_size"),
            num_hidden_layers=spec.get("num_hidden_layers", 24),
            num_attention_heads=spec.get("num_attention_heads", 8),
            num_key_value_heads=spec.get("num_key_value_heads", 4),
            head_dim=spec.get("head_dim", 256),
            global_head_dim=spec.get("global_head_dim"),
            num_experts=spec.get("num_experts", 1),
            num_experts_per_tok=spec.get("num_experts_per_tok", 2),
        )

    @classmethod
    def gemma4_e2b(cls) -> Gemma4Config:
        """Return preset configuration for Gemma 4 E2B matching the JAX specification.

        Returns:
            Configured `Gemma4Config` for E2B.

        """
        return cls._from_spec(GEMMA4_E2B_SPEC)

    @classmethod
    def gemma4_e4b(cls) -> Gemma4Config:
        """Return preset configuration for Gemma 4 E4B matching the JAX specification.

        Returns:
            Configured `Gemma4Config` for E4B.

        """
        return cls._from_spec(GEMMA4_E4B_SPEC)

    @classmethod
    def gemma4_26b_a4b(cls) -> Gemma4Config:
        """Return preset configuration for Gemma 4 26B-A4B (MoE) matching the JAX specification.

        Returns:
            Configured `Gemma4Config` for 26B-A4B.

        """
        return cls._from_spec(GEMMA4_26B_A4B_SPEC)

    @classmethod
    def gemma4_31b(cls) -> Gemma4Config:
        """Return preset configuration for Gemma 4 31B matching the JAX specification.

        Returns:
            Configured `Gemma4Config` for 31B.

        """
        return cls._from_spec(GEMMA4_31B_SPEC)

    @classmethod
    def gemma4_tiny(cls) -> Gemma4Config:
        """Return miniature preset configuration for Gemma 4 Tiny.

        Returns:
            Configured `Gemma4Config` for Tiny.

        """
        return cls._from_spec(GEMMA4_TINY_SPEC)

    @classmethod
    def from_preset(cls, model_name: str) -> Gemma4Config:
        """Resolve a `Gemma4Config` preset from a model identifier string.

        Args:
            model_name: Target model name (e.g., `'google/gemma-4-E2B'`, `'gemma-4-E4B'`, `'gemma-4-26B-A4B'`, `'gemma-4-31B'`, `'gemma-4-tiny'`).

        Returns:
            Matching `Gemma4Config` preset instance.

        """
        key = resolve_gemma4_preset_name(model_name)
        if key == "tiny":
            return cls.gemma4_tiny()
        if key == "31b":
            return cls.gemma4_31b()
        if key == "26b_a4b":
            return cls.gemma4_26b_a4b()
        if key == "e4b":
            return cls.gemma4_e4b()
        return cls.gemma4_e2b()

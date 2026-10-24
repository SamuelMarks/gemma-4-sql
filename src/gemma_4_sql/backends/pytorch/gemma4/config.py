"""Configuration classes and architectural presets for the Gemma 4 PyTorch implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gemma_4_sql.backends.common_gemma4_config import (
    GEMMA4_26B_A4B_SPEC,
    GEMMA4_31B_SPEC,
    GEMMA4_E2B_SPEC,
    GEMMA4_E4B_SPEC,
    resolve_gemma4_preset_name,
)


@dataclass
class Gemma4VisionConfig:
    """Configuration for Gemma 4 Vision sub-model."""

    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    patch_size: int = 14
    image_size: int = 224
    layer_norm_eps: float = 1e-06
    num_channels: int = 3

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
        **kwargs: Any,
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
        for key, value in kwargs.items():
            setattr(self, key, value)


@dataclass
class Gemma4AudioConfig:
    """Configuration for Gemma 4 Audio sub-model."""

    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    rms_norm_eps: float = 1e-06

    def __init__(
        self,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        rms_norm_eps: float = 1e-06,
        **kwargs: Any,
    ) -> None:
        """Initialize the Gemma4AudioConfig.

        Args:
            hidden_size: Audio encoder hidden dimension.
            num_hidden_layers: Number of audio conformer layers.
            num_attention_heads: Number of audio attention heads.
            rms_norm_eps: RMSNorm epsilon constant.
            **kwargs: Additional configuration attributes.

        """
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.rms_norm_eps = rms_norm_eps
        for key, value in kwargs.items():
            setattr(self, key, value)


@dataclass
class Gemma4Config:
    """Configuration for native PyTorch Gemma 4."""

    vocab_size: int = 256000
    hidden_size: int = 2048
    num_hidden_layers: int = 18
    num_attention_heads: int = 8
    num_key_value_heads: int = 1
    intermediate_size: int = 16384
    moe_intermediate_size: int | None = None
    rms_norm_eps: float = 1e-6
    head_dim: int = 256
    global_head_dim: int | None = 512
    pad_token_id: int = 0
    eos_token_id: int | None = None
    image_token_id: int = 255999
    audio_token_id: int = 255998
    num_experts: int = 8
    num_experts_per_tok: int = 2
    router_jitter_noise: float = 0.0
    sliding_window: int | None = 4096
    global_attn_layers: list[int] = field(default_factory=list)
    rope_theta: float = 10000.0
    partial_rotary_factor: float = 1.0
    vision_config: Gemma4VisionConfig = field(default_factory=Gemma4VisionConfig)
    audio_config: Gemma4AudioConfig = field(default_factory=Gemma4AudioConfig)

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
        image_token_id: int = 255999,
        audio_token_id: int = 255998,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        router_jitter_noise: float = 0.0,
        sliding_window: int | None = 4096,
        global_attn_layers: list[int] | None = None,
        rope_theta: float = 10000.0,
        partial_rotary_factor: float = 1.0,
        vision_config: dict[str, Any] | Gemma4VisionConfig | None = None,
        audio_config: dict[str, Any] | Gemma4AudioConfig | None = None,
        moe_intermediate_size: int | None = None,
        global_head_dim: int | None = 512,
        eos_token_id: int | None = None,
        **kwargs: Any,
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
            **kwargs: Additional configuration attributes.

        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
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
        self.router_jitter_noise = router_jitter_noise

        self.sliding_window = sliding_window
        self.global_attn_layers = global_attn_layers or []

        self.rope_theta = rope_theta
        self.partial_rotary_factor = partial_rotary_factor

        if vision_config is None:
            self.vision_config = Gemma4VisionConfig()
        elif isinstance(vision_config, Gemma4VisionConfig):
            self.vision_config = vision_config
        else:
            self.vision_config = Gemma4VisionConfig(**vision_config)

        if audio_config is None:
            self.audio_config = Gemma4AudioConfig()
        elif isinstance(audio_config, Gemma4AudioConfig):
            self.audio_config = audio_config
        else:
            self.audio_config = Gemma4AudioConfig(**audio_config)

        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def gemma4_e2b(cls) -> Gemma4Config:
        """Return preset configuration for Gemma 4 E2B matching the JAX specification.

        Returns:
            Configured `Gemma4Config` for E2B.

        """
        return cls(**GEMMA4_E2B_SPEC)

    @classmethod
    def gemma4_e4b(cls) -> Gemma4Config:
        """Return preset configuration for Gemma 4 E4B matching the JAX specification.

        Returns:
            Configured `Gemma4Config` for E4B.

        """
        return cls(**GEMMA4_E4B_SPEC)

    @classmethod
    def gemma4_26b_a4b(cls) -> Gemma4Config:
        """Return preset configuration for Gemma 4 26B-A4B (MoE) matching the JAX specification.

        Returns:
            Configured `Gemma4Config` for 26B-A4B.

        """
        return cls(**GEMMA4_26B_A4B_SPEC)

    @classmethod
    def gemma4_31b(cls) -> Gemma4Config:
        """Return preset configuration for Gemma 4 31B matching the JAX specification.

        Returns:
            Configured `Gemma4Config` for 31B.

        """
        return cls(**GEMMA4_31B_SPEC)

    @classmethod
    def from_preset(cls, model_name: str) -> Gemma4Config:
        """Resolve a `Gemma4Config` preset from a model identifier string.

        Args:
            model_name: Target model name (e.g., `'google/gemma-4-E2B'`, `'gemma-4-E4B'`, `'gemma-4-26B-A4B'`, `'gemma-4-31B'`).

        Returns:
            Matching `Gemma4Config` preset instance.

        """
        key = resolve_gemma4_preset_name(model_name)
        if key == "31b":
            return cls.gemma4_31b()
        if key == "26b_a4b":
            return cls.gemma4_26b_a4b()
        if key == "e4b":
            return cls.gemma4_e4b()
        return cls.gemma4_e2b()

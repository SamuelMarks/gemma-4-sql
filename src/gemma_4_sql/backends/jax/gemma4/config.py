"""Core functionality for the config module."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import jax.numpy as jnp
from jax.sharding import PartitionSpec

DEFAULT_GRADIENT_CLIPPING = 10000000000.0


@dataclass(frozen=True)
class VisionShardConfig:
    """Sharding configuration for Vision Transformer."""

    attn_kernel: PartitionSpec | None = None
    attn_bias: PartitionSpec | None = None
    attn_qk_activation: PartitionSpec | None = None
    fc1_kernel: PartitionSpec | None = None
    fc1_bias: PartitionSpec | None = None
    fc2_kernel: PartitionSpec | None = None
    fc2_bias: PartitionSpec | None = None
    activation: PartitionSpec | None = None
    layer_norm: PartitionSpec | None = None
    emb_patch_kernel: PartitionSpec | None = None
    emb_patch_bias: PartitionSpec | None = None
    emb_patch_activation: PartitionSpec | None = None
    emb_pos_kernel: PartitionSpec | None = None
    emb_pos_activation: PartitionSpec | None = None

    @staticmethod
    def no_sharding() -> object:
        """Return an unpartitioned default VisionShardConfig.

        Returns:
            The execution result.
        """
        return VisionShardConfig()


@dataclass(frozen=True)
class VisionConfig:
    """Configuration for the Vision Transformer in Gemma 4."""

    hidden_size: int = 1152
    image_size: int = 896
    intermediate_size: int = 4304
    layer_norm_eps: float = 1e-06
    num_attention_heads: int = 16
    num_channels: int = 3
    num_hidden_layers: int = 27
    patch_size: int = 14
    shd_cfg: VisionShardConfig = field(default_factory=VisionShardConfig.no_sharding)


class AttentionType(Enum):
    """Types of attention layers in Gemma 4."""

    LOCAL_SLIDING = "local_sliding"
    GLOBAL = "global"


class ShardMode(Enum):
    """Sharding mode choices."""

    FSDP = "fsdp"
    TP = "tp"


@dataclass(frozen=True)
class ShardConfig:
    """Sharding configuration mappings."""

    attn_kernel: PartitionSpec | None = None
    attn_bias: PartitionSpec | None = None
    attn_qk_activation: PartitionSpec | None = None
    fc1_kernel: PartitionSpec | None = None
    fc1_bias: PartitionSpec | None = None
    fc2_kernel: PartitionSpec | None = None
    fc2_bias: PartitionSpec | None = None
    moe_fc1_kernel: PartitionSpec | None = None
    moe_fc2_kernel: PartitionSpec | None = None
    activation: PartitionSpec | None = None
    norm: PartitionSpec | None = None
    emb_kernel: PartitionSpec | None = None
    cache: PartitionSpec | None = None

    @staticmethod
    def no_sharding() -> object:
        """Return empty sharding config.

        Returns:
            object: The resulting output from the operation.

        """
        return ShardConfig()

    @staticmethod
    def default(*, use_fsdp: bool, use_tp: bool) -> object:
        """Return standard sharding patterns.

        Returns:
            object: The resulting output from the operation.

        """
        fsdp = ShardMode.FSDP.value if use_fsdp else None
        tp = ShardMode.TP.value if use_tp else None
        return ShardConfig(
            attn_kernel=PartitionSpec(tp, fsdp),
            attn_bias=PartitionSpec(tp),
            attn_qk_activation=PartitionSpec(fsdp, tp),
            fc1_kernel=PartitionSpec(fsdp, tp),
            fc1_bias=PartitionSpec(tp),
            fc2_kernel=PartitionSpec(tp, fsdp),
            fc2_bias=PartitionSpec(tp),
            moe_fc1_kernel=PartitionSpec(fsdp, None, tp),
            moe_fc2_kernel=PartitionSpec(fsdp, tp, None),
            activation=PartitionSpec(fsdp, None, tp),
            norm=PartitionSpec(tp),
            emb_kernel=PartitionSpec(None, tp),
            cache=PartitionSpec(fsdp, None, tp, None),
        )


@dataclass(frozen=True)
class AudioConfig:
    """Configuration for the Audio Encoder in Gemma 4."""

    hidden_size: int = 1024
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    hidden_act: str = "silu"
    subsampling_conv_channels: tuple[int, int] = (128, 32)
    conv_kernel_size: int = 5
    residual_weight: float = 0.5
    attention_chunk_size: int = 12
    attention_context_left: int = 13
    attention_context_right: int = 0
    attention_logit_cap: float = 50.0
    attention_invalid_logits_value: float = 1e-09
    use_clipped_linears: bool = True
    gradient_clipping: float = DEFAULT_GRADIENT_CLIPPING
    output_proj_dims: int = 1536
    rms_norm_eps: float = 1e-06


class ModelConfigPresets:
    """Preset configurations for Gemma 4."""

    @classmethod
    def gemma4_base(cls, *, use_fsdp: bool = False, use_tp: bool = False) -> object:
        """Preset configuration for a base Gemma 4 model.

        Returns:
            object: The resulting output from the operation.

        """
        kwargs = {}
        if use_fsdp or use_tp:
            kwargs["shd_cfg"] = ShardConfig.default(use_fsdp=use_fsdp, use_tp=use_tp)
        return cls(**kwargs)

    @classmethod
    def gemma4_e2b(cls, *, use_fsdp: bool = False, use_tp: bool = False) -> object:
        """Preset configuration for Gemma 4 E2B.

        Returns:
            object: The resulting output from the operation.

        """
        kwargs = {}
        if use_fsdp or use_tp:
            kwargs["shd_cfg"] = ShardConfig.default(use_fsdp=use_fsdp, use_tp=use_tp)
        return cls(num_hidden_layers=35, hidden_size=1024, intermediate_size=4096, num_attention_heads=8, num_key_value_heads=4, head_dim=256, global_head_dim=512, num_experts=1, vocab_size=262144, **kwargs)

    @classmethod
    def gemma4_e4b(cls, *, use_fsdp: bool = False, use_tp: bool = False) -> object:
        """Preset configuration for Gemma 4 E4B.

        Returns:
            object: The resulting output from the operation.

        """
        kwargs = {}
        if use_fsdp or use_tp:
            kwargs["shd_cfg"] = ShardConfig.default(use_fsdp=use_fsdp, use_tp=use_tp)
        return cls(num_hidden_layers=42, hidden_size=2560, intermediate_size=10240, num_attention_heads=10, num_key_value_heads=1, head_dim=256, global_head_dim=512, num_experts=1, vocab_size=262144, **kwargs)

    @classmethod
    def gemma4_26b_a4b(cls, *, use_fsdp: bool = False, use_tp: bool = False) -> object:
        """Preset configuration for Gemma 4 26B A4B (MoE).

        Returns:
            object: The resulting output from the operation.

        """
        kwargs = {}
        if use_fsdp or use_tp:
            kwargs["shd_cfg"] = ShardConfig.default(use_fsdp=use_fsdp, use_tp=use_tp)
        return cls(num_hidden_layers=30, hidden_size=2816, intermediate_size=2112, moe_intermediate_size=704, num_attention_heads=8, num_key_value_heads=4, head_dim=256, global_head_dim=512, num_experts=128, num_experts_per_tok=2, vocab_size=262144, **kwargs)

    @classmethod
    def gemma4_31b(cls, *, use_fsdp: bool = False, use_tp: bool = False) -> object:
        """Preset configuration for Gemma 4 31B.

        Returns:
            object: The resulting output from the operation.

        """
        kwargs = {}
        if use_fsdp or use_tp:
            kwargs["shd_cfg"] = ShardConfig.default(use_fsdp=use_fsdp, use_tp=use_tp)
        return cls(num_hidden_layers=60, hidden_size=5376, intermediate_size=21504, num_attention_heads=32, num_key_value_heads=16, head_dim=256, global_head_dim=512, num_experts=1, vocab_size=262144, **kwargs)


@dataclass
class ModelConfig(ModelConfigPresets):
    """Configuration for Gemma 4."""

    vocab_size: int = 256000
    vocab_size_per_layer_input: int | None = None
    hidden_size: int = 2048
    hidden_size_per_layer_input: int | None = None
    intermediate_size: int = 8192
    moe_intermediate_size: int | None = None
    num_hidden_layers: int = 24
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    num_global_key_value_heads: int | None = None
    head_dim: int = 256
    global_head_dim: int | None = None
    rms_norm_eps: float = 1e-06
    sliding_window_size: int = 512
    share_kv_projections: bool = False
    num_experts: int = 4
    num_experts_per_tok: int = 2
    num_shared_experts: int = 1
    dtype: jnp.dtype = jnp.float32
    weight_dtype: jnp.dtype = jnp.float32
    rope_max_timescale: int = 10000
    global_rope_max_timescale: int = 1000000
    local_rope_max_timescale: int | None = None
    local_rope_proportion: float = 1.0
    global_rope_proportion: float = 0.25
    float32_gate_logits: bool = True
    final_logit_softcapping: float | None = None
    attn_logits_soft_cap: float | None = 50.0
    shd_cfg: ShardConfig = field(default_factory=ShardConfig.no_sharding)
    vision_config: VisionConfig | None = None
    audio_config: AudioConfig | None = None
    mm_tokens_per_image: int = 256
    audio_token_id: int | None = None

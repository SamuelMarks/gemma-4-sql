"""Configuration dataclasses and presets for the JAX/Flax NNX Gemma 4 implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import jax.numpy as jnp
from jax.sharding import PartitionSpec

from gemma_4_sql.backends.common_gemma4_config import (
    GEMMA4_26B_A4B_SPEC,
    GEMMA4_31B_SPEC,
    GEMMA4_E2B_SPEC,
    GEMMA4_E4B_SPEC,
    resolve_gemma4_preset_name,
)

DEFAULT_GRADIENT_CLIPPING = 10000000000.0


@dataclass(frozen=True)
class VisionShardConfig:
    """Sharding configuration for the SigLIP Vision Transformer."""

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
    def no_sharding() -> VisionShardConfig:
        """Return an unpartitioned default VisionShardConfig.

        Returns:
            Unpartitioned `VisionShardConfig` instance.

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


def _make_spec(*partitions: str | None) -> PartitionSpec:
    """Instantiate PartitionSpec in a typed context.

    Args:
        *partitions: Partition axis names or None.

    Returns:
        Created `PartitionSpec` instance.

    """
    spec_cls = PartitionSpec
    return spec_cls(*partitions)


@dataclass(frozen=True)
class ShardConfig:
    """Sharding configuration mappings for Gemma 4 layers."""

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
    def no_sharding() -> ShardConfig:
        """Return an empty unpartitioned sharding config.

        Returns:
            Unpartitioned `ShardConfig` instance.

        """
        return ShardConfig()

    @staticmethod
    def default(*, use_fsdp: bool = False, use_tp: bool = False) -> ShardConfig:
        """Return standard FSDP and tensor-parallel sharding patterns.

        Args:
            use_fsdp: Whether to shard across the FSDP axis.
            use_tp: Whether to shard across the tensor-parallel axis.

        Returns:
            Configured `ShardConfig` instance.

        """
        fsdp = ShardMode.FSDP.value if use_fsdp else None
        tp = ShardMode.TP.value if use_tp else None
        return ShardConfig(
            attn_kernel=_make_spec(tp, fsdp),
            attn_bias=_make_spec(tp),
            attn_qk_activation=_make_spec(fsdp, tp),
            fc1_kernel=_make_spec(fsdp, tp),
            fc1_bias=_make_spec(tp),
            fc2_kernel=_make_spec(tp, fsdp),
            fc2_bias=_make_spec(tp),
            moe_fc1_kernel=_make_spec(fsdp, None, tp),
            moe_fc2_kernel=_make_spec(fsdp, tp, None),
            activation=_make_spec(fsdp, None, tp),
            norm=_make_spec(tp),
            emb_kernel=_make_spec(None, tp),
            cache=_make_spec(fsdp, None, tp, None),
        )


@dataclass(frozen=True)
class AudioConfig:
    """Configuration for the Conformer Audio Encoder in Gemma 4."""

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
    """Preset configurations for Gemma 4 variants."""

    @classmethod
    def gemma4_base(cls, *, use_fsdp: bool = False, use_tp: bool = False) -> ModelConfig:
        """Return preset configuration for a base Gemma 4 model.

        Args:
            use_fsdp: Whether to enable FSDP sharding.
            use_tp: Whether to enable tensor-parallel sharding.

        Returns:
            Base `ModelConfig` instance.

        """
        kwargs: dict[str, Any] = {}
        if use_fsdp or use_tp:
            kwargs["shd_cfg"] = ShardConfig.default(use_fsdp=use_fsdp, use_tp=use_tp)
        return ModelConfig(**kwargs)

    @classmethod
    def gemma4_e2b(cls, *, use_fsdp: bool = False, use_tp: bool = False) -> ModelConfig:
        """Return preset configuration for Gemma 4 E2B.

        Args:
            use_fsdp: Whether to enable FSDP sharding.
            use_tp: Whether to enable tensor-parallel sharding.

        Returns:
            Gemma 4 E2B `ModelConfig` instance.

        """
        kwargs: dict[str, Any] = dict(GEMMA4_E2B_SPEC)
        if use_fsdp or use_tp:
            kwargs["shd_cfg"] = ShardConfig.default(use_fsdp=use_fsdp, use_tp=use_tp)
        return ModelConfig(**kwargs)

    @classmethod
    def gemma4_e4b(cls, *, use_fsdp: bool = False, use_tp: bool = False) -> ModelConfig:
        """Return preset configuration for Gemma 4 E4B.

        Args:
            use_fsdp: Whether to enable FSDP sharding.
            use_tp: Whether to enable tensor-parallel sharding.

        Returns:
            Gemma 4 E4B `ModelConfig` instance.

        """
        kwargs: dict[str, Any] = dict(GEMMA4_E4B_SPEC)
        if use_fsdp or use_tp:
            kwargs["shd_cfg"] = ShardConfig.default(use_fsdp=use_fsdp, use_tp=use_tp)
        return ModelConfig(**kwargs)

    @classmethod
    def gemma4_26b_a4b(cls, *, use_fsdp: bool = False, use_tp: bool = False) -> ModelConfig:
        """Return preset configuration for Gemma 4 26B A4B (Mixture-of-Experts).

        Args:
            use_fsdp: Whether to enable FSDP sharding.
            use_tp: Whether to enable tensor-parallel sharding.

        Returns:
            Gemma 4 26B-A4B `ModelConfig` instance.

        """
        kwargs: dict[str, Any] = dict(GEMMA4_26B_A4B_SPEC)
        if use_fsdp or use_tp:
            kwargs["shd_cfg"] = ShardConfig.default(use_fsdp=use_fsdp, use_tp=use_tp)
        return ModelConfig(**kwargs)

    @classmethod
    def gemma4_31b(cls, *, use_fsdp: bool = False, use_tp: bool = False) -> ModelConfig:
        """Return preset configuration for Gemma 4 31B.

        Args:
            use_fsdp: Whether to enable FSDP sharding.
            use_tp: Whether to enable tensor-parallel sharding.

        Returns:
            Gemma 4 31B `ModelConfig` instance.

        """
        kwargs: dict[str, Any] = dict(GEMMA4_31B_SPEC)
        if use_fsdp or use_tp:
            kwargs["shd_cfg"] = ShardConfig.default(use_fsdp=use_fsdp, use_tp=use_tp)
        return ModelConfig(**kwargs)

    @classmethod
    def from_preset(cls, model_name: str, *, use_fsdp: bool = False, use_tp: bool = False) -> ModelConfig:
        """Resolve a `ModelConfig` preset from a model identifier string.

        Args:
            model_name: Target model name (e.g., `'google/gemma-4-E2B'`, `'gemma-4-E4B'`, `'gemma-4-26B-A4B'`, `'gemma-4-31B'`).
            use_fsdp: Whether to enable FSDP sharding.
            use_tp: Whether to enable tensor-parallel sharding.

        Returns:
            Matching `ModelConfig` preset instance.

        """
        preset_kw = {"use_fsdp": use_fsdp, "use_tp": use_tp} if (use_fsdp or use_tp) else {}
        key = resolve_gemma4_preset_name(model_name)
        if key == "31b":
            return cls.gemma4_31b(**preset_kw)
        if key == "26b_a4b":
            return cls.gemma4_26b_a4b(**preset_kw)
        if key == "e4b":
            return cls.gemma4_e4b(**preset_kw)
        return cls.gemma4_e2b(**preset_kw)


@dataclass
class ModelConfig(ModelConfigPresets):
    """Architectural configuration for Gemma 4 in JAX/Flax NNX."""

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
    dtype: jnp.dtype[Any] = jnp.float32
    weight_dtype: jnp.dtype[Any] = jnp.float32
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

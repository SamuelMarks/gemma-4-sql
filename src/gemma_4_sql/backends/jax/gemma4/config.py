"""Configuration dataclasses and presets for the JAX/Flax NNX Gemma 4 implementation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

import jax.numpy as jnp
from jax.sharding import PartitionSpec

from gemma_4_sql.backends.common_gemma4_config import (
    DEFAULT_GRADIENT_CLIPPING,
    GEMMA4_26B_A4B_SPEC,
    GEMMA4_31B_SPEC,
    GEMMA4_E2B_SPEC,
    GEMMA4_E4B_SPEC,
    GEMMA4_TINY_SPEC,
    AttentionType,
    BaseAudioConfigFields,
    BaseGemma4ConfigFields,
    BaseVisionConfigFields,
    resolve_gemma4_preset_name,
)

type JaxDTypeLike = jnp.dtype | type[jnp.generic] | str


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


@dataclass
class VisionConfig(BaseVisionConfigFields):
    """Configuration for the Vision Transformer in Gemma 4."""

    pooling_kernel_size: int = 1
    shd_cfg: VisionShardConfig = field(default_factory=VisionShardConfig.no_sharding)


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


@dataclass
class AudioConfig(BaseAudioConfigFields):
    """Configuration for the Conformer Audio Encoder in Gemma 4."""


class ModelConfigPresets:
    """Preset configurations for Gemma 4 variants."""

    @classmethod
    def _build_from_spec(
        cls,
        spec: Mapping[str, int] | None = None,
        *,
        use_fsdp: bool = False,
        use_tp: bool = False,
    ) -> ModelConfig:
        """Construct a `ModelConfig` from a preset specification and sharding flags.

        Args:
            spec: Optional mapping of integer architectural hyperparameters.
            use_fsdp: Whether to enable FSDP sharding.
            use_tp: Whether to enable tensor-parallel sharding.

        Returns:
            Constructed `ModelConfig` instance.

        """
        shd = ShardConfig.default(use_fsdp=use_fsdp, use_tp=use_tp) if (use_fsdp or use_tp) else ShardConfig.no_sharding()
        if spec is None:
            return ModelConfig(shd_cfg=shd)
        return ModelConfig(
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
            shd_cfg=shd,
        )

    @classmethod
    def gemma4_base(cls, *, use_fsdp: bool = False, use_tp: bool = False) -> ModelConfig:
        """Return preset configuration for a base Gemma 4 model.

        Args:
            use_fsdp: Whether to enable FSDP sharding.
            use_tp: Whether to enable tensor-parallel sharding.

        Returns:
            Base `ModelConfig` instance.

        """
        return cls._build_from_spec(None, use_fsdp=use_fsdp, use_tp=use_tp)

    @classmethod
    def gemma4_e2b(cls, *, use_fsdp: bool = False, use_tp: bool = False) -> ModelConfig:
        """Return preset configuration for Gemma 4 E2B.

        Args:
            use_fsdp: Whether to enable FSDP sharding.
            use_tp: Whether to enable tensor-parallel sharding.

        Returns:
            Gemma 4 E2B `ModelConfig` instance.

        """
        return cls._build_from_spec(GEMMA4_E2B_SPEC, use_fsdp=use_fsdp, use_tp=use_tp)

    @classmethod
    def gemma4_e4b(cls, *, use_fsdp: bool = False, use_tp: bool = False) -> ModelConfig:
        """Return preset configuration for Gemma 4 E4B.

        Args:
            use_fsdp: Whether to enable FSDP sharding.
            use_tp: Whether to enable tensor-parallel sharding.

        Returns:
            Gemma 4 E4B `ModelConfig` instance.

        """
        return cls._build_from_spec(GEMMA4_E4B_SPEC, use_fsdp=use_fsdp, use_tp=use_tp)

    @classmethod
    def gemma4_26b_a4b(cls, *, use_fsdp: bool = False, use_tp: bool = False) -> ModelConfig:
        """Return preset configuration for Gemma 4 26B A4B (Mixture-of-Experts).

        Args:
            use_fsdp: Whether to enable FSDP sharding.
            use_tp: Whether to enable tensor-parallel sharding.

        Returns:
            Gemma 4 26B-A4B `ModelConfig` instance.

        """
        return cls._build_from_spec(GEMMA4_26B_A4B_SPEC, use_fsdp=use_fsdp, use_tp=use_tp)

    @classmethod
    def gemma4_31b(cls, *, use_fsdp: bool = False, use_tp: bool = False) -> ModelConfig:
        """Return preset configuration for Gemma 4 31B.

        Args:
            use_fsdp: Whether to enable FSDP sharding.
            use_tp: Whether to enable tensor-parallel sharding.

        Returns:
            Gemma 4 31B `ModelConfig` instance.

        """
        return cls._build_from_spec(GEMMA4_31B_SPEC, use_fsdp=use_fsdp, use_tp=use_tp)

    @classmethod
    def gemma4_tiny(cls, *, use_fsdp: bool = False, use_tp: bool = False) -> ModelConfig:
        """Return miniature preset configuration for Gemma 4 Tiny.

        Args:
            use_fsdp: Whether to enable FSDP sharding.
            use_tp: Whether to enable tensor-parallel sharding.

        Returns:
            Gemma 4 Tiny `ModelConfig` instance.

        """
        return cls._build_from_spec(GEMMA4_TINY_SPEC, use_fsdp=use_fsdp, use_tp=use_tp)

    @classmethod
    def from_preset(cls, model_name: str, *, use_fsdp: bool = False, use_tp: bool = False) -> ModelConfig:
        """Resolve a `ModelConfig` preset from a model identifier string.

        Args:
            model_name: Target model name (e.g., `'google/gemma-4-E2B'`, `'gemma-4-E4B'`, `'gemma-4-26B-A4B'`, `'gemma-4-31B'`, `'gemma-4-tiny'`).
            use_fsdp: Whether to enable FSDP sharding.
            use_tp: Whether to enable tensor-parallel sharding.

        Returns:
            Matching `ModelConfig` preset instance.

        """
        key = resolve_gemma4_preset_name(model_name)
        kwargs: dict[str, bool] = {}
        if use_fsdp:
            kwargs["use_fsdp"] = use_fsdp
        if use_tp:
            kwargs["use_tp"] = use_tp
        if key == "tiny":
            return cls.gemma4_tiny(**kwargs)
        if key == "31b":
            return cls.gemma4_31b(**kwargs)
        if key == "26b_a4b":
            return cls.gemma4_26b_a4b(**kwargs)
        if key == "e4b":
            return cls.gemma4_e4b(**kwargs)
        return cls.gemma4_e2b(**kwargs)


@runtime_checkable
class _PresetFactoryProtocol(Protocol):
    """Protocol for config classes supporting `from_preset`."""

    @classmethod
    def from_preset(cls, model_name: str) -> ModelConfig:
        """Resolve preset by model name."""
        ...


@runtime_checkable
class _E2BFactoryProtocol(Protocol):
    """Protocol for config classes supporting `gemma4_e2b`."""

    @classmethod
    def gemma4_e2b(cls) -> ModelConfig:
        """Return E2B preset."""
        ...


def resolve_gemma4_config(
    config_cls: type[ModelConfig] | _PresetFactoryProtocol | _E2BFactoryProtocol,
    model_name: str,
) -> ModelConfig:
    """Resolve a `Gemma4Config` preset using `from_preset` or `gemma4_e2b` fallback.

    Args:
        config_cls: `Gemma4Config` class or preset factory.
        model_name: Target model name or preset identifier.

    Returns:
        Resolved configuration instance.

    """
    if isinstance(config_cls, _PresetFactoryProtocol):
        return config_cls.from_preset(model_name)
    return config_cls.gemma4_e2b()


@dataclass
class ModelConfig(BaseGemma4ConfigFields, ModelConfigPresets):
    """Architectural configuration for Gemma 4 in JAX/Flax NNX."""

    dtype: JaxDTypeLike = jnp.float32
    weight_dtype: JaxDTypeLike = jnp.float32
    shd_cfg: ShardConfig = field(default_factory=ShardConfig.no_sharding)
    vision_config: VisionConfig | None = None
    audio_config: AudioConfig | None = None


__all__ = [
    "DEFAULT_GRADIENT_CLIPPING",
    "AttentionType",
    "AudioConfig",
    "JaxDTypeLike",
    "ModelConfig",
    "ModelConfigPresets",
    "ShardConfig",
    "ShardMode",
    "VisionConfig",
    "VisionShardConfig",
    "resolve_gemma4_config",
]

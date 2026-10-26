"""Hardware-agnostic Gemma 4 architectural preset specifications shared across backends."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class AttentionType(Enum):
    """Types of attention layers in Gemma 4."""

    LOCAL_SLIDING = "local_sliding"
    GLOBAL = "global"


GEMMA4_ATTENTION_PATTERN: tuple[AttentionType, ...] = (
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.GLOBAL,
)


@dataclass
class BaseGemma4ConfigFields:
    """Shared hardware-agnostic Gemma 4 architectural hyperparameters."""

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
    num_kv_shared_layers: int = 0
    use_post_norms: bool = True
    rope_max_timescale: int = 10000
    global_rope_max_timescale: int = 1000000
    local_rope_max_timescale: int | None = None
    local_rope_proportion: float = 1.0
    global_rope_proportion: float = 0.25
    float32_gate_logits: bool = True
    final_logit_softcapping: float | None = None
    attn_logits_soft_cap: float | None = 50.0
    mm_tokens_per_image: int = 256
    audio_token_id: int | None = None


GEMMA4_E2B_SPEC: dict[str, Any] = {
    "num_hidden_layers": 35,
    "hidden_size": 1024,
    "intermediate_size": 4096,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "global_head_dim": 512,
    "num_experts": 1,
    "vocab_size": 262144,
}

GEMMA4_E4B_SPEC: dict[str, Any] = {
    "num_hidden_layers": 42,
    "hidden_size": 2560,
    "intermediate_size": 10240,
    "num_attention_heads": 10,
    "num_key_value_heads": 1,
    "head_dim": 256,
    "global_head_dim": 512,
    "num_experts": 1,
    "vocab_size": 262144,
}

GEMMA4_26B_A4B_SPEC: dict[str, Any] = {
    "num_hidden_layers": 30,
    "hidden_size": 2816,
    "intermediate_size": 2112,
    "moe_intermediate_size": 704,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "global_head_dim": 512,
    "num_experts": 128,
    "num_experts_per_tok": 2,
    "vocab_size": 262144,
}

GEMMA4_31B_SPEC: dict[str, Any] = {
    "num_hidden_layers": 60,
    "hidden_size": 5376,
    "intermediate_size": 21504,
    "num_attention_heads": 32,
    "num_key_value_heads": 16,
    "head_dim": 256,
    "global_head_dim": 512,
    "num_experts": 1,
    "vocab_size": 262144,
}

GEMMA4_TINY_SPEC: dict[str, Any] = {
    "num_hidden_layers": 1,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 32,
    "global_head_dim": 32,
    "num_experts": 1,
    "vocab_size": 128,
}


def resolve_gemma4_preset_name(model_name: str) -> str:
    """Resolve the canonical Gemma 4 preset variant key from a model identifier string.

    Args:
        model_name: Target model name (e.g., `'google/gemma-4-E2B'`, `'gemma-4-31B'`, `'gemma-4-tiny'`).

    Returns:
        Canonical preset key (`'tiny'`, `'e2b'`, `'e4b'`, `'26b_a4b'`, or `'31b'`).

    """
    lowered = model_name.lower()
    if "tiny" in lowered or lowered.startswith(("test", "model")):
        return "tiny"
    if "31b" in lowered:
        return "31b"
    if "26b" in lowered or "a4b" in lowered:
        return "26b_a4b"
    if "e4b" in lowered:
        return "e4b"
    return "e2b"


def get_gemma4_preset_spec(model_name: str) -> dict[str, Any]:
    """Return a copy of the architectural hyperparameter dictionary for `model_name`.

    Args:
        model_name: Target model name or preset identifier.

    Returns:
        Dictionary of architectural hyperparameters for the resolved Gemma 4 variant.

    """
    key = resolve_gemma4_preset_name(model_name)
    specs = {
        "tiny": GEMMA4_TINY_SPEC,
        "e2b": GEMMA4_E2B_SPEC,
        "e4b": GEMMA4_E4B_SPEC,
        "26b_a4b": GEMMA4_26B_A4B_SPEC,
        "31b": GEMMA4_31B_SPEC,
    }
    return dict(specs[key])


__all__ = [
    "GEMMA4_26B_A4B_SPEC",
    "GEMMA4_31B_SPEC",
    "GEMMA4_ATTENTION_PATTERN",
    "GEMMA4_E2B_SPEC",
    "GEMMA4_E4B_SPEC",
    "GEMMA4_TINY_SPEC",
    "AttentionType",
    "BaseGemma4ConfigFields",
    "get_gemma4_preset_spec",
    "resolve_gemma4_preset_name",
]

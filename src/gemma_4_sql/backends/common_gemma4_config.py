"""Hardware-agnostic Gemma 4 architectural preset specifications shared across backends."""

from __future__ import annotations

from typing import Any

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


def resolve_gemma4_preset_name(model_name: str) -> str:
    """Resolve the canonical Gemma 4 preset variant key from a model identifier string.

    Args:
        model_name: Target model name (e.g., `'google/gemma-4-E2B'`, `'gemma-4-31B'`).

    Returns:
        Canonical preset key (`'e2b'`, `'e4b'`, `'26b_a4b'`, or `'31b'`).

    """
    lowered = model_name.lower()
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
        "e2b": GEMMA4_E2B_SPEC,
        "e4b": GEMMA4_E4B_SPEC,
        "26b_a4b": GEMMA4_26B_A4B_SPEC,
        "31b": GEMMA4_31B_SPEC,
    }
    return dict(specs[key])


__all__ = [
    "GEMMA4_26B_A4B_SPEC",
    "GEMMA4_31B_SPEC",
    "GEMMA4_E2B_SPEC",
    "GEMMA4_E4B_SPEC",
    "get_gemma4_preset_spec",
    "resolve_gemma4_preset_name",
]

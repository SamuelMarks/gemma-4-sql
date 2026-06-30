"""Tests for JAX PEFT."""

import gemma_4_sql.backends.jax.peft as pt


def test_apply_lora_jax_mocked() -> object:  # type: ignore[return]
    """Initialize function test_apply_lora_jax_mocked."""
    res = pt.apply_lora("test-model", ["q_proj"], 8, 16, 0.05)
    if not res["backend"] == "jax":
        raise AssertionError


def test_apply_lora_jax_real(monkeypatch: object) -> object:  # type: ignore[return]
    """Initialize function test_apply_lora_jax_real.

    Args:
    ----
    monkeypatch: Description of monkeypatch.

    """
    monkeypatch.setattr(pt, "optax", True)  # type: ignore[attr-defined]
    pt.apply_lora("test-model", ["q_proj"], 8, 16, 0.05)

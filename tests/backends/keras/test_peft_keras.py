"""Tests for Keras PEFT."""

import gemma_4_sql.backends.keras.peft as pt


def test_apply_lora_keras_mocked() -> object:  # type: ignore[return]
    """Initialize function test_apply_lora_keras_mocked."""
    res = pt.apply_lora("test-model", ["q_proj"], 8, 16, 0.05)
    if not res["status"] == "completed":
        raise AssertionError
    if not res["backend"] == "keras":
        raise AssertionError


def test_apply_lora_keras_real(monkeypatch: object) -> object:  # type: ignore[return]
    """Initialize function test_apply_lora_keras_real.

    Args:
    ----
    monkeypatch: Description of monkeypatch.

    """
    monkeypatch.setattr(pt, "keras", True)  # type: ignore[attr-defined]
    res = pt.apply_lora("test-model", ["q_proj"], 8, 16, 0.05)
    if not res["status"] == "completed":
        raise AssertionError

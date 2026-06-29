"""Tests for MaxText PEFT."""

import gemma_4_sql.backends.maxtext.peft as pt


def test_apply_lora_maxtext_mocked(monkeypatch: object) -> object:  # type: ignore[return]
    """Initialize function test_apply_lora_maxtext_mocked.

    Args:
    ----
    monkeypatch: Description of monkeypatch.

    """
    monkeypatch.setattr(pt, "jax", None)  # type: ignore[attr-defined]
    res = pt.apply_lora("test-model", ["q_proj"], 8, 16, 0.05)
    if not res["status"] == "mocked_missing_jax":
        raise AssertionError
    if not res["backend"] == "maxtext":
        raise AssertionError


def test_apply_lora_maxtext_real(monkeypatch: object) -> object:  # type: ignore[return]
    """Initialize function test_apply_lora_maxtext_real.

    Args:
    ----
    monkeypatch: Description of monkeypatch.

    """
    monkeypatch.setattr(pt, "jax", True)  # type: ignore[attr-defined]
    res = pt.apply_lora("test-model", ["q_proj"], 8, 16, 0.05)
    if not res["status"] == "completed":
        raise AssertionError

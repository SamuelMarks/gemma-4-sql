"""Tests for PyTorch PEFT."""

import gemma_4_sql.backends.pytorch.peft as pt


def test_apply_lora_pytorch_mocked() -> object:  # type: ignore[return]
    """Initialize function test_apply_lora_pytorch_mocked."""
    res = pt.apply_lora("test-model", ["q_proj"], 8, 16, 0.05)
    if not res["status"] == "mocked_missing_peft":
        raise AssertionError
    if not res["backend"] == "pytorch":
        raise AssertionError


def test_apply_lora_pytorch_real(monkeypatch: object) -> object:  # type: ignore[return]
    """Initialize function test_apply_lora_pytorch_real.

    Args:
    ----
    monkeypatch: Description of monkeypatch.

    """
    monkeypatch.setattr(pt, "peft", True)  # type: ignore[attr-defined]
    res = pt.apply_lora("test-model", ["q_proj"], 8, 16, 0.05)
    if not res["status"] == "completed":
        raise AssertionError

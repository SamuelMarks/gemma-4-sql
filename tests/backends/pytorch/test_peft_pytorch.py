"""Tests for PyTorch PEFT."""

import gemma_4_sql.backends.pytorch.peft as pt


def test_apply_lora_pytorch_mocked():
    res = pt.apply_lora("test-model", ["q_proj"], 8, 16, 0.05)
    assert res["status"] == "mocked_missing_peft"
    assert res["backend"] == "pytorch"


def test_apply_lora_pytorch_real(monkeypatch):
    monkeypatch.setattr(pt, "peft", True)
    res = pt.apply_lora("test-model", ["q_proj"], 8, 16, 0.05)
    assert res["status"] == "completed"

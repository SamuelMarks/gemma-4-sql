"""Tests for MaxText PEFT."""

import gemma_4_sql.backends.maxtext.peft as pt


def test_apply_lora_maxtext_mocked(monkeypatch):
    monkeypatch.setattr(pt, "jax", None)
    res = pt.apply_lora("test-model", ["q_proj"], 8, 16, 0.05)
    assert res["status"] == "mocked_missing_jax"
    assert res["backend"] == "maxtext"


def test_apply_lora_maxtext_real(monkeypatch):
    monkeypatch.setattr(pt, "jax", True)
    res = pt.apply_lora("test-model", ["q_proj"], 8, 16, 0.05)
    assert res["status"] == "completed"

"""Tests for Keras PEFT."""

import gemma_4_sql.backends.keras.peft as pt


def test_apply_lora_keras_mocked():
    res = pt.apply_lora("test-model", ["q_proj"], 8, 16, 0.05)
    assert res["status"] == "mocked_missing_keras"
    assert res["backend"] == "keras"


def test_apply_lora_keras_real(monkeypatch):
    monkeypatch.setattr(pt, "keras", True)
    res = pt.apply_lora("test-model", ["q_proj"], 8, 16, 0.05)
    assert res["status"] == "completed"

"""Tests for JAX PEFT."""

import gemma_4_sql.backends.jax.peft as pt


def test_apply_lora_jax_mocked():
    res = pt.apply_lora("test-model", ["q_proj"], 8, 16, 0.05)
    assert res["status"] == "mocked_missing_optax"
    assert res["backend"] == "jax"


def test_apply_lora_jax_real(monkeypatch):
    monkeypatch.setattr(pt, "optax", True)
    res = pt.apply_lora("test-model", ["q_proj"], 8, 16, 0.05)
    assert res["status"] == "completed"

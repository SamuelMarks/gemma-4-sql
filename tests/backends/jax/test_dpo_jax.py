"""
Tests for JAX DPO logic.
"""

from __future__ import annotations

from typing import Any
import pytest

from gemma_4_sql.backends.jax.dpo import dpo_loss, run_dpo


class MockArray:
    def __sub__(self, other: Any) -> MockArray:
        return MockArray()

    def __mul__(self, other: Any) -> MockArray:
        return MockArray()

    def __rmul__(self, other: Any) -> MockArray:
        return MockArray()

    def __neg__(self) -> MockArray:
        return MockArray()

    def item(self) -> float:
        return 0.42


class MockJnp:
    def array(self, x: Any) -> MockArray:
        return MockArray()

    def mean(self, x: Any) -> MockArray:
        return MockArray()


class MockJnn:
    def log_sigmoid(self, x: Any) -> MockArray:
        return MockArray()


def test_run_dpo_jax_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test JAX DPO when missing."""
    import gemma_4_sql.backends.jax.dpo as jax_dpo

    monkeypatch.setattr(jax_dpo, "jnp", None)
    monkeypatch.setattr(jax_dpo, "jnn", None)

    res = run_dpo("model", "data")
    assert res["status"] == "mocked_missing_jax"
    assert res["final_loss"] == 0.0

    loss, ch_r, re_r = dpo_loss(None, None, None, None)
    assert loss == 0.0
    assert ch_r == 0.0
    assert re_r == 0.0


def test_run_dpo_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test JAX DPO."""
    import gemma_4_sql.backends.jax.dpo as jax_dpo

    monkeypatch.setattr(jax_dpo, "jnp", MockJnp())
    monkeypatch.setattr(jax_dpo, "jnn", MockJnn())

    res = run_dpo("model", "data")
    assert res["backend"] == "jax"
    assert res["status"] == "completed"
    assert res["final_loss"] == 0.42

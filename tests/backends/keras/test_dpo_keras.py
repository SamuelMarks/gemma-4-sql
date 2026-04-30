"""
Tests for Keras DPO logic.
"""

from __future__ import annotations

from typing import Any
import pytest

from gemma_4_sql.backends.keras.dpo import dpo_loss, run_dpo


class MockTensor:
    def __sub__(self, other: Any) -> MockTensor:
        return MockTensor()

    def __mul__(self, other: Any) -> MockTensor:
        return MockTensor()

    def __rmul__(self, other: Any) -> MockTensor:
        return MockTensor()

    def __neg__(self) -> MockTensor:
        return MockTensor()

    def numpy(self) -> float:
        return 0.42


class MockMath:
    def log_sigmoid(self, x: Any) -> MockTensor:
        return MockTensor()


class MockTf:
    float32 = "float32"

    def __init__(self):
        self.math = MockMath()

    def constant(self, x: Any, dtype: Any = None) -> MockTensor:
        return MockTensor()

    def reduce_mean(self, x: Any) -> MockTensor:
        return MockTensor()


def test_run_dpo_keras_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Keras DPO when missing."""
    import gemma_4_sql.backends.keras.dpo as keras_dpo

    monkeypatch.setattr(keras_dpo, "tf", None)

    res = run_dpo("model", "data")
    assert res["status"] == "mocked_missing_keras"
    assert res["final_loss"] == 0.0

    loss, ch_r, re_r = dpo_loss(None, None, None, None)
    assert loss == 0.0
    assert ch_r == 0.0
    assert re_r == 0.0


def test_run_dpo_keras(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Keras DPO."""
    import gemma_4_sql.backends.keras.dpo as keras_dpo

    monkeypatch.setattr(keras_dpo, "tf", MockTf())

    res = run_dpo("model", "data")
    assert res["backend"] == "keras"
    assert res["status"] == "completed"
    assert res["final_loss"] == 0.42

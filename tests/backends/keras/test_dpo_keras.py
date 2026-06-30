"""Tests for Keras DPO logic."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemma_4_sql.backends.keras.dpo import dpo_loss, run_dpo

if TYPE_CHECKING:
    import pytest


class MockTensor:
    """Initialize class MockTensor."""

    def __sub__(self: object, other: object) -> MockTensor:
        """Initialize function __sub__.

        Args:
        ----
        other: Description of other.

        """
        return MockTensor()

    def __mul__(self: object, other: object) -> MockTensor:
        """Initialize function __mul__.

        Args:
        ----
        other: Description of other.

        """
        return MockTensor()

    def __rmul__(self: object, other: object) -> MockTensor:
        """Initialize function __rmul__.

        Args:
        ----
        other: Description of other.

        """
        return MockTensor()

    def __neg__(self: object) -> MockTensor:
        """Initialize function __neg__."""
        return MockTensor()

    def numpy(self: object) -> float:
        """Initialize function numpy."""
        return 0.42


class MockMath:
    """Initialize class MockMath."""

    def log_sigmoid(self: object, _x: object) -> MockTensor:
        """Initialize function log_sigmoid.

        Args:
        ----
        x: Description of x.

        """
        return MockTensor()


class MockTf:
    """Initialize class MockTf."""

    float32 = "float32"

    def __init__(self: object) -> None:
        """Initialize function __init__."""
        self.math = MockMath()

    def constant(self: object, *_args: object, **_kwargs: object) -> MockTensor:
        """Initialize function constant.

        Args:
        ----
        x: Description of x.
        dtype: Description of dtype.

        """
        return MockTensor()

    def reduce_mean(self: object, _x: object) -> MockTensor:
        """Initialize function reduce_mean.

        Args:
        ----
        x: Description of x.

        """
        return MockTensor()


def test_run_dpo_keras_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Keras DPO when missing."""
    keras_dpo = __import__("gemma_4_sql.backends.keras.dpo", fromlist=[""])
    monkeypatch.setattr(keras_dpo, "tf", None)
    res = run_dpo("model", "data")
    if not res["status"] == "mocked_missing_keras":
        raise AssertionError
    (loss, ch_r, re_r) = dpo_loss(None, None, None, None)
    if not loss == 0.0:
        raise AssertionError
    if not ch_r == 0.0:
        raise AssertionError
    if not re_r == 0.0:
        raise AssertionError


def test_run_dpo_keras(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Keras DPO."""
    keras_dpo = __import__("gemma_4_sql.backends.keras.dpo", fromlist=[""])
    monkeypatch.setattr(keras_dpo, "tf", MockTf())
    res = run_dpo("model", "data")
    if not res["backend"] == "keras":
        raise AssertionError

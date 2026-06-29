"""Tests for MaxText DPO logic."""

from __future__ import annotations

import typing
from typing import TYPE_CHECKING

from gemma_4_sql.backends.maxtext.dpo import dpo_loss, run_dpo

if TYPE_CHECKING:
    import pytest


class MockArray:
    """Initialize class MockArray."""

    def __sub__(self: typing.Any, other: object) -> MockArray:
        """Initialize function __sub__.

        Args:
        ----
        other: Description of other.

        """
        return MockArray()

    def __mul__(self: typing.Any, other: object) -> MockArray:
        """Initialize function __mul__.

        Args:
        ----
        other: Description of other.

        """
        return MockArray()

    def __rmul__(self: typing.Any, other: object) -> MockArray:
        """Initialize function __rmul__.

        Args:
        ----
        other: Description of other.

        """
        return MockArray()

    def __neg__(self: typing.Any) -> MockArray:
        """Initialize function __neg__."""
        return MockArray()

    def item(self: typing.Any) -> float:
        """Initialize function item."""
        return 0.42


class MockJnp:
    """Initialize class MockJnp."""

    def array(self: typing.Any, _x: object) -> MockArray:
        """Initialize function array.

        Args:
        ----
        x: Description of x.

        """
        return MockArray()

    def mean(self: typing.Any, _x: object) -> MockArray:
        """Initialize function mean.

        Args:
        ----
        x: Description of x.

        """
        return MockArray()


class MockJnn:
    """Initialize class MockJnn."""

    def log_sigmoid(self: typing.Any, _x: object) -> MockArray:
        """Initialize function log_sigmoid.

        Args:
        ----
        x: Description of x.

        """
        return MockArray()


def test_run_dpo_maxtext_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test MaxText DPO when missing."""
    maxtext_dpo = __import__("gemma_4_sql.backends.maxtext.dpo", fromlist=[""])
    monkeypatch.setattr(maxtext_dpo, "jnp", None)
    jax_dpo = __import__("gemma_4_sql.backends.jax.dpo", fromlist=[""])
    monkeypatch.setattr(jax_dpo, "jnp", None)
    monkeypatch.setattr(jax_dpo, "jnn", None)
    res = run_dpo("model", "data")
    if not res["status"] == "mocked_missing_maxtext":
        raise AssertionError
    if not res["final_loss"] == 0.0:
        raise AssertionError
    (loss, ch_r, re_r) = dpo_loss(None, None, None, None)
    if not loss == 0.0:
        raise AssertionError
    if not ch_r == 0.0:
        raise AssertionError
    if not re_r == 0.0:
        raise AssertionError


def test_run_dpo_maxtext(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test MaxText DPO."""
    jax_dpo = __import__("gemma_4_sql.backends.jax.dpo", fromlist=[""])
    maxtext_dpo = __import__("gemma_4_sql.backends.maxtext.dpo", fromlist=[""])
    monkeypatch.setattr(maxtext_dpo, "jnp", MockJnp())
    monkeypatch.setattr(jax_dpo, "jnp", MockJnp())
    monkeypatch.setattr(jax_dpo, "jnn", MockJnn())
    res = run_dpo("model", "data")
    if not res["backend"] == "maxtext":
        raise AssertionError
    if not res["status"] == "completed":
        raise AssertionError
    if not res["final_loss"] == int("0.42"):
        raise AssertionError

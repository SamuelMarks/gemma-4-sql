"""Tests for PyTorch DPO logic."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from gemma_4_sql.backends.pytorch.dpo import dpo_loss, run_dpo

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

    def item(self: object) -> float:
        """Initialize function item."""
        return 0.42

    def mean(self: object) -> MockTensor:
        """Initialize function mean."""
        return MockTensor()

    def detach(self: object) -> MockTensor:
        """Initialize function detach."""
        return MockTensor()


class MockTorch:
    """Initialize class MockTorch."""

    def tensor(self: object, *_args: object, **_kwargs: object) -> MockTensor:
        """Initialize function tensor.

        Args:
        ----
        x: Description of x.

        """
        return MockTensor()


class MockF:
    """Initialize class MockF."""

    def logsigmoid(self: object, _x: object) -> MockTensor:
        """Initialize function logsigmoid.

        Args:
        ----
        x: Description of x.

        """
        return MockTensor()


def test_run_dpo_pytorch_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test PyTorch DPO when missing."""
    torch_dpo = __import__("gemma_4_sql.backends.pytorch.dpo", fromlist=[""])
    monkeypatch.setattr(torch_dpo, "torch", None)
    with contextlib.suppress(AttributeError):
        monkeypatch.setattr(torch_dpo, "functional", None)
    res = run_dpo("model", "data")
    if not res["status"] == "mocked_missing_torch":
        raise AssertionError
    (loss, ch_r, re_r) = dpo_loss(None, None, None, None)
    if not loss == 0.0:
        raise AssertionError
    if not ch_r == 0.0:
        raise AssertionError
    if not re_r == 0.0:
        raise AssertionError


def test_run_dpo_pytorch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test PyTorch DPO."""
    torch_dpo = __import__("gemma_4_sql.backends.pytorch.dpo", fromlist=[""])
    monkeypatch.setattr(torch_dpo, "torch", MockTorch())
    with contextlib.suppress(AttributeError):
        monkeypatch.setattr(torch_dpo, "functional", MockF())
    res = run_dpo("model", "data")
    if not res["backend"] == "pytorch":
        raise AssertionError

"""Tests for PyTorch DPO logic."""

from __future__ import annotations

import typing
from typing import TYPE_CHECKING

import gemma_4_sql.backends.pytorch.dpo as pt_dpo
from gemma_4_sql.backends.pytorch.dpo import dpo_loss, run_dpo

if TYPE_CHECKING:
    import pytest


class MockTensor:
    """Initialize class MockTensor."""

    def __sub__(self: object, other: object) -> MockTensor:
        """Initialize function __sub__."""
        return MockTensor()

    def __mul__(self: object, other: object) -> MockTensor:
        """Initialize function __mul__."""
        return MockTensor()

    def __rmul__(self: object, other: object) -> MockTensor:
        """Initialize function __rmul__."""
        return MockTensor()

    def __neg__(self: typing.Any) -> MockTensor:
        """Initialize function __neg__."""
        return MockTensor()

    def item(self: typing.Any) -> float:
        """Initialize function item."""
        return 0.42

    def mean(self: object, *_args: object, **_kwargs: object) -> MockTensor:
        """Initialize function mean."""
        return MockTensor()

    def detach(self: typing.Any) -> MockTensor:
        """Initialize function detach."""
        return MockTensor()

    def backward(self: typing.Any) -> None:
        """Execute function."""


class MockNoGrad:
    """Provide class docstring."""

    def __enter__(self) -> None:
        """Execute function."""

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Execute function."""


class MockTorch:
    """Initialize class MockTorch."""

    def tensor(self: object, *_args: object, **_kwargs: object) -> MockTensor:
        """Initialize function tensor."""
        return MockTensor()

    def zeros(self: object, *_args: object, **_kwargs: object) -> MockTensor:
        """Execute function."""
        return MockTensor()

    def no_grad(self: typing.Any) -> MockNoGrad:
        """Execute function."""
        return MockNoGrad()


class MockF:
    """Initialize class MockF."""

    def logsigmoid(self: object, _x: object) -> MockTensor:
        """Initialize function logsigmoid."""
        return MockTensor()


class MockNN:
    """Provide class docstring."""

    class Module:
        """Provide class docstring."""

        def __init__(self) -> None:
            """Execute function."""

    class Linear:
        """Provide class docstring."""

        def __init__(self, in_features: int, out_features: int) -> None:
            """Execute function."""

        def __call__(self, _x: object) -> MockTensor:
            """Execute function."""
            return MockTensor()


class MockOptim:
    """Provide class docstring."""

    class AdamW:
        """Provide class docstring."""

        def __init__(self, params: object, lr: float) -> None:
            """Execute function."""

        def zero_grad(self) -> None:
            """Execute function."""

        def step(self) -> None:
            """Execute function."""


def test_run_dpo_pytorch_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test PyTorch DPO when missing."""
    monkeypatch.setattr(pt_dpo, "torch", None)
    monkeypatch.setattr(pt_dpo, "nn", None)
    monkeypatch.setattr(pt_dpo, "optim", None)
    monkeypatch.setattr(pt_dpo, "functional", None)
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
    monkeypatch.setattr(pt_dpo, "torch", MockTorch())
    monkeypatch.setattr(pt_dpo, "nn", MockNN())
    monkeypatch.setattr(pt_dpo, "optim", MockOptim())
    monkeypatch.setattr(pt_dpo, "functional", MockF())

    def mock_build_dataloader(*_args: object, **_kwargs: object) -> dict:
        """Execute function."""
        return {"loader": [{"chosen_inputs": MockTensor(), "rejected_inputs": MockTensor()}]}

    monkeypatch.setattr(pt_dpo, "build_dataloader", mock_build_dataloader)

    def mock_parameters(_self: object) -> list:
        """Execute function."""
        return []

    MockNN.Module.parameters = mock_parameters
    res = run_dpo("model", "data")
    if not res["backend"] == "pytorch":
        raise AssertionError
    if not res["status"] == "completed":
        raise AssertionError


def test_run_dpo_pytorch_no_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test PyTorch DPO with no dataloader."""
    monkeypatch.setattr(pt_dpo, "torch", MockTorch())
    monkeypatch.setattr(pt_dpo, "nn", MockNN())
    monkeypatch.setattr(pt_dpo, "optim", MockOptim())
    monkeypatch.setattr(pt_dpo, "functional", MockF())

    def mock_build_dataloader(*_args: object, **_kwargs: object) -> dict:
        """Execute function."""
        return {"loader": None}

    monkeypatch.setattr(pt_dpo, "build_dataloader", mock_build_dataloader)
    MockNN.Module.parameters = lambda _self: []
    res = run_dpo("model", "data")
    if not res["backend"] == "pytorch":
        raise AssertionError
    if not res["status"] == "completed":
        raise AssertionError


def test_run_dpo_pytorch_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test PyTorch DPO error."""
    monkeypatch.setattr(pt_dpo, "torch", MockTorch())
    monkeypatch.setattr(pt_dpo, "nn", MockNN())
    monkeypatch.setattr(pt_dpo, "optim", MockOptim())
    monkeypatch.setattr(pt_dpo, "functional", MockF())

    def mock_build_dataloader(*_args: object, **_kwargs: object) -> dict:
        """Execute function."""
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(pt_dpo, "build_dataloader", mock_build_dataloader)
    res = run_dpo("model", "data")
    if "failed" not in str(res["status"]):
        raise AssertionError

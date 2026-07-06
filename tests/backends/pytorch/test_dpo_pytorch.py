"""Tests for PyTorch DPO logic."""

from __future__ import annotations

import typing

import pytest

import gemma_4_sql.backends.pytorch.dpo as pt_dpo
from gemma_4_sql.backends.pytorch.dpo import dpo_loss, run_dpo
from gemma_4_sql.type_hints import DPOConfig


class MockTensor:
    """Initialize class MockTensor."""

    def __sub__(self: object, other: object) -> MockTensor:
        """Initialize function __sub__.

        Returns:
            object: Description of return.

        """
        return MockTensor()

    def __mul__(self: object, other: object) -> MockTensor:
        """Initialize function __mul__.

        Returns:
            object: Description of return.

        """
        return MockTensor()

    def __rmul__(self: object, other: object) -> MockTensor:
        """Initialize function __rmul__.

        Returns:
            object: Description of return.

        """
        return MockTensor()

    def __neg__(self: typing.Any) -> MockTensor:
        """Initialize function __neg__.

        Returns:
            object: Description of return.

        """
        return MockTensor()

    def item(self: typing.Any) -> float:
        """Initialize function item.

        Returns:
            object: Description of return.

        """
        return 0.42

    def mean(self: object, *_args: object, **_kwargs: object) -> MockTensor:
        """Initialize function mean.

        Returns:
            object: Description of return.

        """
        return MockTensor()

    def detach(self: typing.Any) -> MockTensor:
        """Initialize function detach.

        Returns:
            object: Description of return.

        """
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
        """Initialize function tensor.

        Returns:
            object: Description of return.

        """
        return MockTensor()

    def zeros(self: object, *_args: object, **_kwargs: object) -> MockTensor:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return MockTensor()

    def no_grad(self: typing.Any) -> MockNoGrad:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return MockNoGrad()


class MockF:
    """Initialize class MockF."""

    def logsigmoid(self: object, _x: object) -> MockTensor:
        """Initialize function logsigmoid.

        Returns:
            object: Description of return.

        """
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
            """Execute function.

            Returns:
                object: Description of return.

            """
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
    """Test PyTorch DPO when missing.

    Raises:
        AssertionError: Description.

    """
    from gemma_4_sql.exceptions import DependencyMissingError

    monkeypatch.setattr(pt_dpo, "torch", None)
    monkeypatch.setattr(pt_dpo, "nn", None)
    monkeypatch.setattr(pt_dpo, "optim", None)
    monkeypatch.setattr(pt_dpo, "functional", None)
    with pytest.raises(DependencyMissingError, match="PyTorch dependencies are missing."):
        run_dpo(DPOConfig(model_name="model", dataset="data"))
    (loss, ch_r, re_r) = dpo_loss(None, None, None, None)
    if not loss == 0.0:
        raise AssertionError
    if not ch_r == 0.0:
        raise AssertionError
    if not re_r == 0.0:
        raise AssertionError


def _mock_transformers_import(monkeypatch: pytest.MonkeyPatch) -> None:
    builtins = __import__("builtins", fromlist=[""])
    orig_import = builtins.__import__

    class MockGemma4Instance:
        def parameters(self):
            return []

        def __call__(self, _x, **kwargs):
            return MockTensor()

    class MockGemma4:
        @classmethod
        def from_pretrained(cls, *_args: object, **_kwargs: object) -> object:
            return MockGemma4Instance()

    def mock_import(name: object, _globals: object = None, _locals: object = None, fromlist: object = (), level: object = 0) -> object:
        if name == "transformers.models.gemma4" and "Gemma4ForCausalLM" in fromlist:
            return type("M", (), {"Gemma4ForCausalLM": MockGemma4})
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", mock_import)


def test_run_dpo_pytorch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test PyTorch DPO.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(pt_dpo, "torch", MockTorch())
    monkeypatch.setattr(pt_dpo, "nn", MockNN())
    monkeypatch.setattr(pt_dpo, "optim", MockOptim())
    monkeypatch.setattr(pt_dpo, "functional", MockF())
    _mock_transformers_import(monkeypatch)

    def mock_build_dataloader(*_args: object, **_kwargs: object) -> dict:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return {"loader": [{"chosen_inputs": MockTensor(), "rejected_inputs": MockTensor()}]}

    monkeypatch.setattr(pt_dpo, "build_dataloader", mock_build_dataloader)

    def mock_parameters(_self: object) -> list:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return []

    MockNN.Module.parameters = mock_parameters
    res = run_dpo(DPOConfig(model_name="model", dataset="data"))
    if not res["backend"] == "pytorch":
        raise AssertionError
    if not res["status"] == "completed":
        raise AssertionError


def test_run_dpo_pytorch_no_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test PyTorch DPO with no dataloader.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(pt_dpo, "torch", MockTorch())
    monkeypatch.setattr(pt_dpo, "nn", MockNN())
    monkeypatch.setattr(pt_dpo, "optim", MockOptim())
    monkeypatch.setattr(pt_dpo, "functional", MockF())
    _mock_transformers_import(monkeypatch)

    def mock_build_dataloader(*_args: object, **_kwargs: object) -> dict:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return {"loader": None}

    monkeypatch.setattr(pt_dpo, "build_dataloader", mock_build_dataloader)
    MockNN.Module.parameters = lambda _self: []
    res = run_dpo(DPOConfig(model_name="model", dataset="data"))
    if not res["backend"] == "pytorch":
        raise AssertionError
    if not res["status"] == "completed":
        raise AssertionError


def test_run_dpo_pytorch_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test PyTorch DPO error.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(pt_dpo, "torch", MockTorch())
    monkeypatch.setattr(pt_dpo, "nn", MockNN())
    monkeypatch.setattr(pt_dpo, "optim", MockOptim())
    monkeypatch.setattr(pt_dpo, "functional", MockF())
    _mock_transformers_import(monkeypatch)

    def mock_build_dataloader(*_args: object, **_kwargs: object) -> dict:
        """Execute function.

        Raises:
            ValueError: Description.

        """
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(pt_dpo, "build_dataloader", mock_build_dataloader)
    res = run_dpo(DPOConfig(model_name="model", dataset="data"))
    if "failed" not in str(res["status"]):
        raise AssertionError

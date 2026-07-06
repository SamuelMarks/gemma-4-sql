"""Tests for PyTorch PEFT."""

from __future__ import annotations

from typing import NoReturn as Never

import pytest

import gemma_4_sql.backends.pytorch.peft as pt_peft
from gemma_4_sql.backends.pytorch.peft import apply_lora


class MockPeft:
    """Provide class docstring."""


class MockTorch:
    """Provide class docstring."""


class MockLoraConfig:
    """Provide class docstring."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Execute function."""


def mock_get_peft_model(model: object, _config: object) -> object:
    """Execute function.

    Returns:
        object: Description of return.

    """
    return model


class MockAutoModelForCausalLM:
    """Provide class docstring."""

    @staticmethod
    def from_pretrained(_model_name: str) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """

        class Model:
            """Provide class docstring."""

            def print_trainable_parameters(self) -> None:
                """Execute function."""

        return Model()


def test_apply_lora_pytorch_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test PyTorch PEFT when missing.

    Raises:
        AssertionError: Description.

    """
    from gemma_4_sql.exceptions import DependencyMissingError

    monkeypatch.setattr(pt_peft, "peft", None)
    monkeypatch.setattr(pt_peft, "torch", None)
    monkeypatch.setattr(pt_peft, "AutoModelForCausalLM", None)
    with pytest.raises(DependencyMissingError, match="PyTorch PEFT dependencies are missing."):
        apply_lora("test-model", ["q_proj"], 8, 16, 0.05)


def test_apply_lora_pytorch_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test PyTorch PEFT.

    Raises:
        AssertionError: Description.

    """
    monkeypatch.setattr(pt_peft, "peft", MockPeft())
    monkeypatch.setattr(pt_peft, "torch", MockTorch())
    monkeypatch.setattr(pt_peft, "LoraConfig", MockLoraConfig)
    monkeypatch.setattr(pt_peft, "get_peft_model", mock_get_peft_model)
    monkeypatch.setattr(pt_peft, "AutoModelForCausalLM", MockAutoModelForCausalLM)
    res = apply_lora("test-model", ["q_proj"], 8, 16, 0.05)
    if not res["status"] == "completed":
        raise AssertionError
    if not res["backend"] == "pytorch":
        raise AssertionError


class ErrorAutoModel:
    """Provide class docstring."""

    def from_pretrained(*args, **kwargs) -> Never:
        """Mock method.

        Raises:
            ValueError: Description.

        """
        msg = "err"
        raise ValueError(msg)


def test_apply_lora_pytorch_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test PyTorch PEFT error."""
    monkeypatch.setattr(pt_peft, "peft", MockPeft())
    monkeypatch.setattr(pt_peft, "torch", MockTorch())
    monkeypatch.setattr(pt_peft, "LoraConfig", MockLoraConfig)
    monkeypatch.setattr(pt_peft, "get_peft_model", mock_get_peft_model)

    def mock_raise_error(_model_name: str) -> object:
        """Execute function.

        Raises:
            ValueError: Description.

        """
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(pt_peft, "AutoModelForCausalLM", ErrorAutoModel)
    res = apply_lora("test-model", ["q_proj"], 8, 16, 0.05)
    assert "failed" in str(res["status"])


def test_peft_imports_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    importlib = __import__("importlib", fromlist=[""])
    sys = __import__("sys", fromlist=[""])
    m_peft = __import__("gemma_4_sql.backends.pytorch.peft", fromlist=[""])
    monkeypatch.setitem(sys.modules, "torch", type("M", (), {})())
    monkeypatch.setitem(sys.modules, "peft", type("M", (), {"LoraConfig": None, "get_peft_model": None})())
    monkeypatch.setitem(sys.modules, "transformers", type("M", (), {"AutoModelForCausalLM": None})())
    importlib.reload(m_peft)
    monkeypatch.undo()
    importlib.reload(m_peft)

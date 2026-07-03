"""Tests for PyTorch PEFT."""

from __future__ import annotations

from typing import TYPE_CHECKING

import gemma_4_sql.backends.pytorch.peft as pt_peft
from gemma_4_sql.backends.pytorch.peft import apply_lora

if TYPE_CHECKING:
    import pytest


class MockPeft:
    """Provide class docstring."""


class MockTorch:
    """Provide class docstring."""


class MockLoraConfig:
    """Provide class docstring."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Execute function."""


def mock_get_peft_model(model: object, _config: object) -> object:
    """Execute function."""
    return model


class MockAutoModelForCausalLM:
    """Provide class docstring."""

    @staticmethod
    def from_pretrained(_model_name: str) -> object:
        """Execute function."""

        class Model:
            """Provide class docstring."""

            def print_trainable_parameters(self) -> None:
                """Execute function."""

        return Model()


def test_apply_lora_pytorch_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test PyTorch PEFT when missing."""
    monkeypatch.setattr(pt_peft, "peft", None)
    monkeypatch.setattr(pt_peft, "torch", None)
    monkeypatch.setattr(pt_peft, "AutoModelForCausalLM", None)
    res = apply_lora("test-model", ["q_proj"], 8, 16, 0.05)
    if not res["status"] == "mocked_missing_peft":
        raise AssertionError
    if not res["backend"] == "pytorch":
        raise AssertionError


def test_apply_lora_pytorch_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test PyTorch PEFT."""
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

    from_pretrained = Exception


def test_apply_lora_pytorch_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test PyTorch PEFT error."""
    monkeypatch.setattr(pt_peft, "peft", MockPeft())
    monkeypatch.setattr(pt_peft, "torch", MockTorch())
    monkeypatch.setattr(pt_peft, "LoraConfig", MockLoraConfig)
    monkeypatch.setattr(pt_peft, "get_peft_model", mock_get_peft_model)

    def mock_raise_error(_model_name: str) -> object:
        """Execute function."""
        msg = "err"
        raise ValueError(msg)

    monkeypatch.setattr(pt_peft, "AutoModelForCausalLM", ErrorAutoModel)
    res = apply_lora("test-model", ["q_proj"], 8, 16, 0.05)
    if "failed" not in str(res["status"]):
        raise AssertionError


def test_peft_imports_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    importlib = __import__("importlib")
    sys = __import__("sys")
    m_peft = __import__("gemma_4_sql.backends.pytorch.peft")
    monkeypatch.setitem(sys.modules, "torch", type("M", (), {})())
    monkeypatch.setitem(sys.modules, "peft", type("M", (), {"LoraConfig": None, "get_peft_model": None})())
    monkeypatch.setitem(sys.modules, "transformers", type("M", (), {"AutoModelForCausalLM": None})())
    importlib.reload(m_peft)
    monkeypatch.undo()
    importlib.reload(m_peft)

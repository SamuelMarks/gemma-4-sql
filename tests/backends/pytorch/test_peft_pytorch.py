"""Tests for PyTorch PEFT."""

from __future__ import annotations

from typing import TYPE_CHECKING

import gemma_4_sql.backends.pytorch.peft as pt_peft
from gemma_4_sql.backends.pytorch.peft import apply_lora

if TYPE_CHECKING:
    import pytest


class MockPeft:
    pass


class MockTorch:
    pass


class MockLoraConfig:
    def __init__(self, *args: object, **kwargs: object) -> None:
        pass


def mock_get_peft_model(model: object, config: object) -> object:
    return model


class MockAutoModelForCausalLM:
    @staticmethod
    def from_pretrained(model_name: str) -> object:
        class Model:
            def print_trainable_parameters(self) -> None:
                pass

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


def test_apply_lora_pytorch_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test PyTorch PEFT error."""
    monkeypatch.setattr(pt_peft, "peft", MockPeft())
    monkeypatch.setattr(pt_peft, "torch", MockTorch())
    monkeypatch.setattr(pt_peft, "LoraConfig", MockLoraConfig)
    monkeypatch.setattr(pt_peft, "get_peft_model", mock_get_peft_model)

    def raise_error(model_name: str) -> object:
        msg = "err"
        raise ValueError(msg)

    class ErrorAutoModel:
        from_pretrained = raise_error

    monkeypatch.setattr(pt_peft, "AutoModelForCausalLM", ErrorAutoModel)

    res = apply_lora("test-model", ["q_proj"], 8, 16, 0.05)
    if "failed" not in str(res["status"]):
        raise AssertionError


def test_peft_imports_success(monkeypatch: pytest.MonkeyPatch):
    import importlib
    import sys

    import gemma_4_sql.backends.pytorch.peft as m_peft

    monkeypatch.setitem(sys.modules, "torch", type("M", (), {})())
    monkeypatch.setitem(sys.modules, "peft", type("M", (), {"LoraConfig": None, "get_peft_model": None})())
    monkeypatch.setitem(sys.modules, "transformers", type("M", (), {"AutoModelForCausalLM": None})())

    importlib.reload(m_peft)
    monkeypatch.undo()
    importlib.reload(m_peft)

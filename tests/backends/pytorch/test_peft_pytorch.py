"""Module docstring."""

from unittest import mock

import pytest
from torch import nn

import gemma_4_sql.backends.pytorch.peft as pt_peft
from gemma_4_sql.backends.pytorch.peft import apply_lora


def test_peft_import_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test importing peft module successfully."""
    import importlib
    import sys

    mock_peft = type("peft", (), {"LoraConfig": "mocked", "get_peft_model": "mocked"})
    mock_transformers = type("transformers", (), {"AutoModelForCausalLM": "mocked"})
    monkeypatch.setitem(sys.modules, "peft", mock_peft)
    monkeypatch.setitem(sys.modules, "transformers", mock_transformers)

    import gemma_4_sql.backends.pytorch.peft as pt_peft_mod

    importlib.reload(pt_peft_mod)
    assert pt_peft_mod.peft is not None
    assert pt_peft_mod.LoraConfig == "mocked"

    del sys.modules["peft"]
    del sys.modules["transformers"]
    importlib.reload(pt_peft_mod)


class DummyModel(nn.Module):
    """Docstring for DummyModel."""

    def __init__(self):
        """Docstring for __init__."""
        super().__init__()
        self.q_proj = nn.Linear(10, 10)
        self.saved_path = None
        self.printed = False

    def print_trainable_parameters(self):
        """Docstring for print_trainable_parameters."""
        self.printed = True

    def save_pretrained(self, path: str):
        """Docstring for save_pretrained."""
        self.saved_path = path

    @classmethod
    def from_pretrained(cls, model_name: str, *args, **kwargs):
        """Docstring for from_pretrained."""
        if "error" in model_name:
            raise ValueError("mock error")
        return cls()


@pytest.fixture
def mock_transformers_auto_model(monkeypatch):
    """Docstring for mock_transformers_auto_model."""
    import builtins

    orig_import = builtins.__import__

    def mock_import(name, *a, **k):
        """Docstring for mock_import."""
        if name == "transformers":
            return type("MockTransformers", (), {"AutoModelForCausalLM": DummyModel})
        return orig_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    monkeypatch.setattr(pt_peft, "AutoModelForCausalLM", DummyModel)


def test_apply_lora_pytorch_real(monkeypatch: pytest.MonkeyPatch, mock_transformers_auto_model: None) -> None:
    """Docstring for test_apply_lora_pytorch_real."""
    monkeypatch.setattr(pt_peft, "peft", mock.MagicMock())
    monkeypatch.setattr(pt_peft, "torch", mock.MagicMock())
    monkeypatch.setattr(pt_peft, "LoraConfig", mock.MagicMock())
    monkeypatch.setattr(pt_peft, "get_peft_model", mock.MagicMock())
    res = apply_lora("test-model", ["q_proj"], 8, 16, 0.05, output_dir="/tmp/peft_adapter")
    assert res["status"] == "completed"
    assert res["backend"] == "pytorch"
    assert res["model"] == "test-model"

    res_no_out = apply_lora("test-model", ["q_proj"], 8, 16, 0.05)
    assert res_no_out["status"] == "completed"


def test_apply_lora_pytorch_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Docstring for test_apply_lora_pytorch_missing_deps."""
    from gemma_4_sql.exceptions import DependencyMissingError

    monkeypatch.setattr(pt_peft, "peft", None)
    with pytest.raises(DependencyMissingError, match="PyTorch PEFT dependencies are missing"):
        apply_lora("test-model", ["q_proj"], 8, 16, 0.05)


def test_apply_lora_pytorch_error(monkeypatch: pytest.MonkeyPatch, mock_transformers_auto_model: None) -> None:
    """Docstring for test_apply_lora_pytorch_error."""
    monkeypatch.setattr(pt_peft, "peft", mock.MagicMock())
    monkeypatch.setattr(pt_peft, "torch", mock.MagicMock())
    monkeypatch.setattr(pt_peft, "LoraConfig", mock.MagicMock())

    def mock_get_peft_model(*args, **kwargs):
        """Docstring for mock_get_peft_model."""
        raise ValueError("mock error")

    monkeypatch.setattr(pt_peft, "get_peft_model", mock_get_peft_model)
    res = apply_lora("error-model", ["q_proj"], 8, 16, 0.05)
    assert "failed" in str(res["status"])

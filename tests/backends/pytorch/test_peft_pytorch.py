"""Tests for PyTorch peft."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

import gemma_4_sql.backends.pytorch.peft as peft_module
from gemma_4_sql.exceptions import DependencyMissingError, TrainingError
from tests.conftest import requires_torch

pytestmark = requires_torch


def test_apply_lora_real_tiny_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify PyTorch apply_lora injects LoRA adapters into a real tiny Gemma4ForCausalLM."""
    from gemma_4_sql.backends.pytorch.gemma4.config import Gemma4Config
    from gemma_4_sql.backends.pytorch.gemma4.modeling import Gemma4ForCausalLM

    cfg = Gemma4Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=2,
        num_experts_per_tok=1,
    )
    tiny_model = Gemma4ForCausalLM(cfg)

    mock_auto = MagicMock()
    mock_auto.from_pretrained.return_value = tiny_model
    monkeypatch.setattr(peft_module, "AutoModelForCausalLM", mock_auto)
    monkeypatch.setattr(peft_module, "peft", MagicMock())
    monkeypatch.setattr(peft_module, "LoraConfig", MagicMock(return_value="lora_cfg"))
    monkeypatch.setattr(peft_module, "get_peft_model", lambda model, _cfg: model)

    res = peft_module.apply_lora("tiny-gemma4", ["q_proj", "v_proj"], lora_r=4, lora_alpha=8)
    assert res["status"] == "completed"
    assert res["lora_r"] == 4
    assert res["injected_modules"] >= 2

    class BadNamedMods:
        """Stub raising RuntimeError from named_modules."""

        def named_modules(self) -> object:
            """Raise RuntimeError."""
            raise RuntimeError("boom")

    assert peft_module._count_injected_peft_modules(BadNamedMods(), ["q_proj"]) == 1
    assert peft_module._count_injected_peft_modules(object(), ["q_proj", "v_proj"]) == 2


def test_apply_lora(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test apply_lora branches with monkeypatched dependencies."""
    monkeypatch.setattr(peft_module, "peft", MagicMock())
    monkeypatch.setattr(peft_module, "torch", MagicMock())

    mock_auto = MagicMock()
    monkeypatch.setattr(peft_module, "AutoModelForCausalLM", mock_auto)

    mock_lora_cfg = MagicMock()
    monkeypatch.setattr(peft_module, "LoraConfig", mock_lora_cfg)

    mock_get_peft = MagicMock()
    monkeypatch.setattr(peft_module, "get_peft_model", mock_get_peft)

    mock_model: Any = MagicMock()
    mock_get_peft.return_value = mock_model

    res = peft_module.apply_lora("m", ["l1", "l2"], output_dir="out")
    assert res["status"] == "completed"
    assert res["injected_modules"] == 2
    mock_model.print_trainable_parameters.assert_called_once()
    mock_model.save_pretrained.assert_called_once_with("out")

    del mock_model.print_trainable_parameters
    del mock_model.save_pretrained
    res = peft_module.apply_lora("m", ["l1", "l2"], output_dir="out")
    assert res["status"] == "completed"

    mock_auto.from_pretrained.side_effect = ValueError("error")
    res = peft_module.apply_lora("m", ["l"])
    assert "failed: error" in res["status"]

    with pytest.raises(TrainingError, match="error"):
        peft_module.apply_lora("m", ["l"], raise_on_error=True)

    monkeypatch.setattr(peft_module, "peft", None)
    with pytest.raises(DependencyMissingError):
        peft_module.apply_lora("m", ["l"])

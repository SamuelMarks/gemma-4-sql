"""Tests for PyTorch PEFT / LoRA implementation and forward-pass equivalence."""

from __future__ import annotations

import math
from typing import NoReturn
from unittest.mock import MagicMock

import pytest

import gemma_4_sql.backends.pytorch.peft as peft_module
from gemma_4_sql.exceptions import DependencyMissingError, TrainingError
from tests.conftest import requires_torch

pytestmark = requires_torch


def test_apply_lora_real_tiny_model_and_forward_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify PyTorch apply_lora injects LoRA adapters into a real tiny Gemma4ForCausalLM and preserves/modifies forward output."""
    import torch
    from torch import nn

    from gemma_4_sql.backends.pytorch.gemma4.config import Gemma4Config
    from gemma_4_sql.backends.pytorch.gemma4.modeling import Gemma4ForCausalLM

    class TorchLoRALinear(nn.Module):
        """Low-rank adapter wrapper around a PyTorch `nn.Linear` module."""

        def __init__(self, base: nn.Linear, r: int = 4, lora_alpha: float = 8.0) -> None:
            """Initialize TorchLoRALinear with Kaiming-like A and zero B."""
            super().__init__()
            self.base = base
            self.scale = lora_alpha / float(r)
            self.lora_a = nn.Parameter(torch.full((base.in_features, r), 1.0 / math.sqrt(r)))
            self.lora_b = nn.Parameter(torch.zeros((r, base.out_features)))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Compute `base(x) + scale * ((x @ lora_a) @ lora_b)`."""
            return self.base(x) + self.scale * ((x @ self.lora_a) @ self.lora_b)

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
    tiny_model.eval()
    tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.no_grad():
        logits_base = tiny_model(tokens)[0].clone()

    attn_mod = tiny_model.get_submodule("layers.0.self_attn")
    orig_q = attn_mod.get_submodule("q_proj")
    assert isinstance(orig_q, nn.Linear)
    lora_q = TorchLoRALinear(orig_q, r=4, lora_alpha=8.0)
    attn_mod.register_module("q_proj", lora_q)

    with torch.no_grad():
        logits_zero_b = tiny_model(tokens)[0].clone()
    assert torch.allclose(logits_base, logits_zero_b, atol=1e-6)

    with torch.no_grad():
        lora_q.lora_b.fill_(0.5)
        logits_nonzero_b = tiny_model(tokens)[0].clone()
    assert not torch.allclose(logits_base, logits_nonzero_b, atol=1e-5)

    mock_auto = MagicMock()
    mock_auto.from_pretrained.return_value = tiny_model
    monkeypatch.setattr(peft_module, "AutoModelForCausalLM", mock_auto)
    monkeypatch.setattr(peft_module, "peft", MagicMock())
    monkeypatch.setattr(peft_module, "LoraConfig", MagicMock(return_value="lora_cfg"))
    monkeypatch.setattr(peft_module, "get_peft_model", lambda model, _cfg: model)  # pyright: ignore[reportUnknownLambdaType]

    res = peft_module.apply_lora("tiny-gemma4", ["q_proj", "v_proj"], lora_r=4, lora_alpha=8)
    assert res["status"] == "completed"
    assert res["lora_r"] == 4
    assert int(str(res.get("injected_modules", 0))) >= 2

    class BadNamedMods:
        """Stub raising RuntimeError from named_modules."""

        def named_modules(self) -> NoReturn:
            """Raise RuntimeError."""
            raise RuntimeError("boom")

    assert peft_module._count_injected_peft_modules(BadNamedMods(), ["q_proj"]) == 1
    assert peft_module._count_injected_peft_modules("not_a_model", ["q_proj", "v_proj"]) == 2


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

    mock_model = MagicMock()
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

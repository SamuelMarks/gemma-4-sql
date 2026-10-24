"""Tests for PyTorch peft."""

from unittest.mock import MagicMock

from tests.conftest import requires_torch

pytestmark = requires_torch

import pytest

from gemma_4_sql.exceptions import DependencyMissingError, TrainingError


@pytest.fixture(autouse=True)
def _restore_peft_mod(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore peft_module globals after each test."""
    import gemma_4_sql.backends.pytorch.peft as peft_module

    monkeypatch.setattr(peft_module, "peft", peft_module.peft)
    monkeypatch.setattr(peft_module, "torch", peft_module.torch)
    monkeypatch.setattr(peft_module, "LoraConfig", peft_module.LoraConfig)
    monkeypatch.setattr(peft_module, "get_peft_model", peft_module.get_peft_model)
    monkeypatch.setattr(peft_module, "AutoModelForCausalLM", peft_module.AutoModelForCausalLM)


def test_apply_lora_with_real_tiny_pytorch_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify apply_lora runs against a real tiny PyTorch Gemma4ForCausalLM instance."""
    import gemma_4_sql.backends.pytorch.peft as peft_module
    from gemma_4_sql.backends.pytorch.gemma4 import Gemma4Config, Gemma4ForCausalLM

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
        layer_types=["sliding_attention"],
        hidden_size_per_layer_input=0,
        num_kv_shared_layers=0,
    )
    tiny_model = Gemma4ForCausalLM(cfg)
    mock_auto = MagicMock()
    mock_auto.from_pretrained.return_value = tiny_model
    monkeypatch.setattr(peft_module, "AutoModelForCausalLM", mock_auto)
    monkeypatch.setattr(peft_module, "peft", MagicMock())
    monkeypatch.setattr(peft_module, "LoraConfig", MagicMock())
    monkeypatch.setattr(peft_module, "get_peft_model", lambda model, _cfg: model)

    res = peft_module.apply_lora("tiny-gemma4", ["q_proj", "v_proj"], lora_r=4, lora_alpha=8)
    assert res["status"] == "completed"
    assert res["lora_r"] == 4


def test_apply_lora():
    """Test apply_lora."""
    import gemma_4_sql.backends.pytorch.peft as peft_module

    peft_module.peft = MagicMock()
    peft_module.torch = MagicMock()
    peft_module.LoraConfig = MagicMock()
    peft_module.get_peft_model = MagicMock()
    peft_module.AutoModelForCausalLM = MagicMock()

    # Test valid execution
    mock_model = MagicMock()
    peft_module.AutoModelForCausalLM.from_pretrained.return_value = mock_model
    peft_module.get_peft_model.return_value = mock_model

    res = peft_module.apply_lora("m", ["l"], output_dir="dir")
    assert res["status"] == "completed"

    mock_model.print_trainable_parameters.assert_called_once()
    mock_model.save_pretrained.assert_called_once_with("dir")

    # Missing hasattr print
    del mock_model.print_trainable_parameters
    res2 = peft_module.apply_lora("m", ["l"], output_dir="dir")
    assert res2["status"] == "completed"

    # Missing kwargs
    del mock_model.save_pretrained
    res3 = peft_module.apply_lora("m", ["l"])
    assert res3["status"] == "completed"

    # Test error during execution
    peft_module.get_peft_model.side_effect = RuntimeError("error")
    res = peft_module.apply_lora("m", ["l"])
    assert "failed: error" in res["status"]
    with pytest.raises(TrainingError, match="error"):
        peft_module.apply_lora("m", ["l"], raise_on_error=True)

    # Test missing dependencies
    peft_module.peft = None
    with pytest.raises(DependencyMissingError):
        peft_module.apply_lora("m", ["l"])

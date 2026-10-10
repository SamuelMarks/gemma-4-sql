"""Tests for PyTorch peft."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.exceptions import DependencyMissingError


def test_pytorch_peft_imports():
    """Test pytorch peft imports fallback."""
    import importlib

    with patch.dict(sys.modules, {"peft": None, "torch": None, "transformers": None}):
        import gemma_4_sql.backends.pytorch.peft as peft_module

        importlib.reload(peft_module)
        assert peft_module.peft is None
        assert peft_module.torch is None
        assert peft_module.LoraConfig is None
        assert peft_module.get_peft_model is None
        assert peft_module.AutoModelForCausalLM is None

    with patch.dict(sys.modules, {"peft": MagicMock(LoraConfig="LoraConfig", get_peft_model="get_peft_model"), "torch": MagicMock(), "transformers": MagicMock(AutoModelForCausalLM="AutoModelForCausalLM")}):
        importlib.reload(peft_module)
        assert peft_module.peft is not None
        assert peft_module.torch is not None

    importlib.reload(peft_module)


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

    # Test missing dependencies
    peft_module.peft = None
    with pytest.raises(DependencyMissingError):
        peft_module.apply_lora("m", ["l"])

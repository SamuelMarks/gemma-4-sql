"""Tests for PyTorch export."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.exceptions import DependencyMissingError, ExportError
from tests.conftest import requires_torch

pytestmark = requires_torch


@pytest.fixture(autouse=True)
def _restore_export_mod(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore export_module globals after each test."""
    import gemma_4_sql.backends.pytorch.export as export_module

    monkeypatch.setattr(export_module, "torch", export_module.torch)
    monkeypatch.setattr(export_module, "save_file", export_module.save_file)


def test_is_rank_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _is_rank_zero under uninitialized, rank-0, and non-rank-0 distributed states."""
    import gemma_4_sql.backends.pytorch.export as export_module

    monkeypatch.setattr(export_module, "torch", None)
    assert export_module._is_rank_zero() is True

    monkeypatch.setattr(export_module, "torch", MagicMock())
    with patch("torch.distributed.is_initialized", return_value=False):
        assert export_module._is_rank_zero() is True

    with patch("torch.distributed.is_initialized", return_value=True), patch("torch.distributed.get_rank", return_value=0):
        assert export_module._is_rank_zero() is True

    with patch("torch.distributed.is_initialized", return_value=True), patch("torch.distributed.get_rank", return_value=1):
        assert export_module._is_rank_zero() is False


def test_save_real_model_and_export_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test _save_real_model and export_model with real tiny Gemma4ForCausalLM and HF mocks."""
    import gemma_4_sql.backends.pytorch.export as export_module
    from gemma_4_sql.backends.pytorch.gemma4.config import Gemma4Config

    export_dir = tmp_path / "export_native"
    tiny_cfg = Gemma4Config(
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

    res_native = export_module.export_model(
        "tiny_native",
        str(export_dir),
        backend_alias="pytorch_native",
        config=tiny_cfg,
    )
    assert res_native["status"] == "exported_with_safetensors"
    assert Path(res_native["file_path"]).exists()

    # Test test_model preset and generic preset branches on pytorch_native
    with patch("gemma_4_sql.backends.pytorch.gemma4.modeling.Gemma4ForCausalLM") as mock_native:
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {"a": MagicMock(), "lm_head.weight": MagicMock()}
        mock_native.return_value = mock_model
        monkeypatch.setattr(export_module, "save_file", MagicMock())
        with patch("gemma_4_sql.backends.pytorch.gemma4.modeling.Gemma4Config"):
            f1, stat1 = export_module._save_real_model("test_model", str(export_dir), backend_alias="pytorch_native")
            assert f1.name == "model.safetensors"
            assert stat1 == "exported_with_safetensors"

            f2, stat2 = export_module._save_real_model("other", str(export_dir), backend_alias="pytorch_native")
            assert f2.name == "model.safetensors"
            assert stat2 == "exported_with_safetensors"

    # Test HF branch via fake transformers.models.gemma4 module in sys.modules
    mock_hf_mod = MagicMock()
    mock_hf_model = MagicMock()
    mock_hf_model.state_dict.return_value = {"a": MagicMock(), "lm_head.weight": MagicMock()}
    mock_hf_mod.Gemma4ForCausalLM.from_pretrained.return_value = mock_hf_model
    monkeypatch.setitem(sys.modules, "transformers.models.gemma4", mock_hf_mod)

    f_hf, stat_hf = export_module._save_real_model("m", str(export_dir), backend_alias="pytorch_hf")
    assert f_hf.name == "model.safetensors"
    assert stat_hf == "exported_with_safetensors"

    _, stat_skip = export_module._save_real_model("m", str(export_dir), is_rank_zero=False, backend_alias="pytorch_hf")
    assert stat_skip == "skipped_non_rank_zero"

    f_adap, _ = export_module._save_real_model("m", str(export_dir), export_type="adapter", backend_alias="pytorch_hf")
    assert f_adap.name == "adapter_model.safetensors"

    monkeypatch.setattr(export_module, "save_file", None)
    _, stat_nosave = export_module._save_real_model("m", str(export_dir), backend_alias="pytorch_hf")
    assert stat_nosave == "exported_with_safetensors"

    mock_hf_mod.Gemma4ForCausalLM.from_pretrained.side_effect = ValueError("error")
    with pytest.raises(ExportError, match="Failed to load model m"):
        export_module._save_real_model("m", str(export_dir), backend_alias="pytorch_hf")

    with pytest.raises(DependencyMissingError):
        export_module.export_model("m", "export")

    monkeypatch.setattr(export_module, "torch", None)
    with pytest.raises(DependencyMissingError):
        export_module.export_model("m", "export")


def test_is_rank_zero_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _is_rank_zero returns True when torch.distributed.is_initialized raises RuntimeError."""
    import gemma_4_sql.backends.pytorch.export as pexp

    monkeypatch.setattr(pexp, "torch", MagicMock())
    with patch("torch.distributed.is_initialized", side_effect=RuntimeError("mocked error")):
        assert pexp._is_rank_zero() is True


def test_is_rank_zero_get_rank_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _is_rank_zero returns True when torch.distributed.get_rank raises RuntimeError."""
    import torch.distributed as dist

    import gemma_4_sql.backends.pytorch.export as pexp

    monkeypatch.setattr(dist, "is_initialized", lambda: True)

    def _raise_rank() -> int:
        """Raise RuntimeError."""
        raise RuntimeError("mocked error")

    monkeypatch.setattr(dist, "get_rank", _raise_rank)
    assert pexp._is_rank_zero() is True

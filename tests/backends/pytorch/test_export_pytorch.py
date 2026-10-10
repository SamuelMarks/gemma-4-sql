"""Tests for PyTorch export."""

import sys
from unittest.mock import MagicMock, patch

import pytest


def test_pytorch_export_imports():
    """Test pytorch export imports fallback."""
    import importlib

    with patch.dict(sys.modules, {"torch": None, "safetensors.torch": None}):
        import gemma_4_sql.backends.pytorch.export as export_module

        importlib.reload(export_module)
        assert export_module.torch is None
        assert export_module.save_file is None
    importlib.reload(export_module)


def test_is_rank_zero():
    """Test _is_rank_zero."""
    import gemma_4_sql.backends.pytorch.export as export_module

    export_module.torch = None
    assert export_module._is_rank_zero() is True

    export_module.torch = MagicMock()
    with patch("builtins.__import__") as mock_import:
        mock_dist = MagicMock()
        mock_dist.is_initialized.return_value = False
        mock_import.return_value = mock_dist

        assert export_module._is_rank_zero() is True

        mock_dist.is_initialized.return_value = True
        mock_dist.get_rank.return_value = 0
        assert export_module._is_rank_zero() is True

        mock_dist.get_rank.return_value = 1
        assert export_module._is_rank_zero() is False

        mock_import.side_effect = ImportError("error")
        assert export_module._is_rank_zero() is True


def test_save_real_model(tmp_path):
    """Test _save_real_model."""
    import gemma_4_sql.backends.pytorch.export as export_module

    export_dir = tmp_path / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    mock_model = MagicMock()
    mock_model.state_dict.return_value = {"a": MagicMock(), "lm_head.weight": MagicMock()}

    export_module.save_file = MagicMock()

    with patch("builtins.__import__") as mock_import:
        mock_cls = MagicMock()
        mock_cls.Gemma4ForCausalLM.from_pretrained.return_value = mock_model
        mock_import.return_value = mock_cls

        # Test HF
        f, stat = export_module._save_real_model("m", str(export_dir), backend_alias="pytorch_hf")
        assert f.name == "model.safetensors"
        assert stat == "exported_with_safetensors"

        # Test HF non-rank zero
        f, stat = export_module._save_real_model("m", str(export_dir), is_rank_zero=False, backend_alias="pytorch_hf")
        assert stat == "skipped_non_rank_zero"

        # Test adapter
        f, stat = export_module._save_real_model("m", str(export_dir), export_type="adapter", backend_alias="pytorch_hf")
        assert f.name == "adapter_model.safetensors"

        # Test HF rank zero but no save_file
        export_module.save_file = None
        f, stat = export_module._save_real_model("m", str(export_dir), backend_alias="pytorch_hf")
        assert stat == "exported_with_safetensors"
        export_module.save_file = MagicMock()

        # Test error
        mock_cls.Gemma4ForCausalLM.from_pretrained.side_effect = ValueError("error")
        with pytest.raises(ValueError):
            export_module._save_real_model("m", str(export_dir), backend_alias="pytorch_hf")

    # Test native
    with patch("gemma_4_sql.backends.pytorch.gemma4.modeling.Gemma4ForCausalLM") as mock_native:
        mock_native.return_value = mock_model
        with patch("gemma_4_sql.backends.pytorch.gemma4.modeling.Gemma4Config"):
            # test config gen
            f, stat = export_module._save_real_model("test_model", str(export_dir), backend_alias="pytorch_native")
            assert f.name == "model.safetensors"

            # test provided config
            f, stat = export_module._save_real_model("m", str(export_dir), backend_alias="pytorch_native", config=MagicMock())

            # test generic config
            f, stat = export_module._save_real_model("other", str(export_dir), backend_alias="pytorch_native")


def test_export_model():
    """Test export_model."""
    import gemma_4_sql.backends.pytorch.export as export_module

    export_module.torch = MagicMock()
    export_module.save_file = MagicMock()

    with patch("gemma_4_sql.backends.pytorch.export._is_rank_zero") as mock_rank:
        mock_rank.return_value = True

        with patch("gemma_4_sql.backends.pytorch.export._save_real_model") as mock_save:
            mock_save.return_value = ("file_path", "ok")

            res = export_module.export_model("m", "export")
            assert res["status"] == "ok"
            assert res["file_path"] == "file_path"

    export_module.torch = None
    with pytest.raises(RuntimeError):
        export_module.export_model("m", "export")

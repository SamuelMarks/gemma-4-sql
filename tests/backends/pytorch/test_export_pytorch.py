from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import gemma_4_sql.backends.pytorch.export as ex


def test_is_rank_zero():
    # Torch missing
    with patch("gemma_4_sql.backends.pytorch.export.torch", None):
        assert ex._is_rank_zero() is True

    mock_torch = MagicMock()
    with patch("gemma_4_sql.backends.pytorch.export.torch", mock_torch):
        # Test not initialized
        mock_dist = MagicMock()
        mock_dist.is_initialized.return_value = False
        with patch("builtins.__import__", return_value=mock_dist):
            assert ex._is_rank_zero() is True

        # Test initialized, rank 0
        mock_dist.is_initialized.return_value = True
        mock_dist.get_rank.return_value = 0
        with patch("builtins.__import__", return_value=mock_dist):
            assert ex._is_rank_zero() is True

        # Test initialized, rank 1
        mock_dist.get_rank.return_value = 1
        with patch("builtins.__import__", return_value=mock_dist):
            assert ex._is_rank_zero() is False

        # Test ImportError
        def mock_import(*args, **kwargs):
            raise ImportError()

        with patch("builtins.__import__", side_effect=mock_import):
            assert ex._is_rank_zero() is True


def test_save_real_model():
    mock_save_file = MagicMock()

    with patch("gemma_4_sql.backends.pytorch.export.save_file", mock_save_file):
        # Test native loading
        class MockConfig:
            def __init__(self, **kwargs):
                pass

        class MockNativeModel:
            def __init__(self, cfg):
                pass

            def state_dict(self):
                mock_t = MagicMock()
                mock_t.clone.return_value = "cloned_weight"
                return {"lm_head.weight": mock_t, "other": "weight"}

        with patch.dict("sys.modules", {"gemma_4_sql.backends.pytorch.gemma4.modeling": MagicMock(Gemma4Config=MockConfig, Gemma4ForCausalLM=MockNativeModel)}):
            file_path, status = ex._save_real_model("model_name", "export_path", is_rank_zero=True, backend_alias="pytorch_native")
            assert str(file_path).endswith("model.safetensors")
            assert status == "exported_with_safetensors"
            # save_file should be called with tensors
            args, _ = mock_save_file.call_args
            tensors, _fp = args
            assert tensors["lm_head.weight"] == "cloned_weight"
            assert tensors["other"] == "weight"

            # Test default config for tests
            file_path, status = ex._save_real_model("test_model", "export_path", is_rank_zero=True, backend_alias="pytorch_native")

            # Test custom config provided
            file_path, status = ex._save_real_model("model_name", "export_path", is_rank_zero=True, backend_alias="pytorch_native", config=MockConfig())

            # Test non-matching model name for config
            file_path, status = ex._save_real_model("dummy_name", "export_path", is_rank_zero=True, backend_alias="pytorch_native")

        # Test hf loading
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {"k": "v"}
        mock_model_cls = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        with patch("builtins.__import__", return_value=MagicMock(Gemma4ForCausalLM=mock_model_cls)):
            file_path, status = ex._save_real_model("model_name", "export_path", is_rank_zero=True, backend_alias="pytorch_hf")
            assert str(file_path).endswith("model.safetensors")

        # Test adapter export
        mock_model.save_pretrained = MagicMock()
        with patch("builtins.__import__", return_value=MagicMock(Gemma4ForCausalLM=mock_model_cls)):
            file_path, status = ex._save_real_model("model_name", "export_path", is_rank_zero=True, backend_alias="pytorch_hf", export_type="adapter")
            assert str(file_path).endswith("adapter_model.safetensors")
            mock_model.save_pretrained.assert_called_with("export_path")

        # Test adapter export without save_pretrained
        mock_model_no_save = MagicMock()
        mock_model_no_save.state_dict.return_value = {"k": "v"}
        del mock_model_no_save.save_pretrained
        mock_model_cls_2 = MagicMock()
        mock_model_cls_2.from_pretrained.return_value = mock_model_no_save
        with patch("builtins.__import__", return_value=MagicMock(Gemma4ForCausalLM=mock_model_cls_2)):
            file_path, status = ex._save_real_model("model_name", "export_path", is_rank_zero=True, backend_alias="pytorch_hf", export_type="adapter")
            assert str(file_path).endswith("model.safetensors")

        # Test loading error
        with patch("builtins.__import__", side_effect=ImportError), pytest.raises(ValueError):
            ex._save_real_model("model", "path", backend_alias="pytorch_hf")

        # Test non-rank zero skipping safetensors saving
        with patch("builtins.__import__", return_value=MagicMock(Gemma4ForCausalLM=mock_model_cls)):
            file_path, status = ex._save_real_model("model_name", "export_path", is_rank_zero=False, backend_alias="pytorch_hf")
            assert status == "skipped_non_rank_zero"


def test_export_model():
    # Test dependencies missing
    with patch("gemma_4_sql.backends.pytorch.export.torch", None), pytest.raises(RuntimeError):
        ex.export_model("m", "p")

    mock_torch = MagicMock()
    with (
        patch("gemma_4_sql.backends.pytorch.export.torch", mock_torch),
        patch("gemma_4_sql.backends.pytorch.export.save_file", MagicMock()),
        patch("pathlib.Path.mkdir"),
        patch("gemma_4_sql.backends.pytorch.export._is_rank_zero", return_value=True),
        patch("gemma_4_sql.backends.pytorch.export._save_real_model") as mock_save,
    ):
        mock_save.return_value = (Path("export_path/model.safetensors"), "ok")

        res = ex.export_model("m", "export_path", backend_alias="alias", other_arg="val")
        assert res["backend"] == "alias"
        assert res["model"] == "m"
        assert res["export_path"] == "export_path"
        assert res["status"] == "ok"
        mock_save.assert_called_with("m", "export_path", is_rank_zero=True, backend_alias="alias", other_arg="val")

import json
from unittest.mock import MagicMock, patch

import pytest

import gemma_4_sql.backends.mlx.export as mlx_export
from gemma_4_sql.backends.mlx.export import export_model


@pytest.fixture
def mock_mx(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr(mlx_export, "mx", mock)
    return mock


def test_export_model_missing_mx(monkeypatch, tmp_path):
    monkeypatch.setattr(mlx_export, "mx", None)
    with pytest.raises(RuntimeError, match="MLX is not installed"):
        export_model("model", str(tmp_path))


def test_export_model_load_failure(mock_mx, tmp_path):
    def mock_import(name, fromlist=None, *args, **kwargs):
        if name == "mlx_lm":
            raise ImportError("Failed to import")
        return __import__(name, fromlist=fromlist, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import), pytest.raises(ValueError, match="Failed to load MLX model"):
        export_model("model", str(tmp_path))


def test_export_model_success_dict_config(mock_mx, tmp_path):
    mock_model = MagicMock()
    mock_model.parameters.return_value = [("layer1", MagicMock())]
    mock_model.config = {"model_type": "gemma"}

    mock_load = MagicMock(return_value=(mock_model, MagicMock()))

    def mock_import(name, fromlist=None, *args, **kwargs):
        if name == "mlx_lm":
            m = MagicMock()
            m.load = mock_load
            return m
        import builtins

        return builtins.__import__(name, fromlist=fromlist, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        res = export_model("model", str(tmp_path))

    assert res["status"] == "exported_with_safetensors"
    mock_mx.save_safetensors.assert_called_once()
    assert (tmp_path / "config.json").exists()

    with open(tmp_path / "config.json") as f:
        cfg = json.load(f)
        assert cfg["model_type"] == "gemma"


def test_export_model_success_object_config(mock_mx, tmp_path):
    mock_model = MagicMock()
    mock_model.parameters.return_value = [("layer1", MagicMock())]

    class Config:
        def __init__(self):
            self.model_type = "gemma2"

    mock_model.config = Config()

    mock_load = MagicMock(return_value=(mock_model, MagicMock()))

    def mock_import(name, fromlist=None, *args, **kwargs):
        if name == "mlx_lm":
            m = MagicMock()
            m.load = mock_load
            return m
        import builtins

        return builtins.__import__(name, fromlist=fromlist, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        res = export_model("model", str(tmp_path))

    assert res["status"] == "exported_with_safetensors"
    with open(tmp_path / "config.json") as f:
        cfg = json.load(f)
        assert cfg["model_type"] == "gemma2"


def test_export_model_success_no_config(mock_mx, tmp_path):
    mock_model = MagicMock()
    mock_model.parameters.return_value = [("layer1", MagicMock())]
    del mock_model.config

    mock_load = MagicMock(return_value=(mock_model, MagicMock()))

    def mock_import(name, fromlist=None, *args, **kwargs):
        if name == "mlx_lm":
            m = MagicMock()
            m.load = mock_load
            return m
        import builtins

        return builtins.__import__(name, fromlist=fromlist, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        res = export_model("my_model", str(tmp_path))

    assert res["status"] == "exported_with_safetensors"
    with open(tmp_path / "config.json") as f:
        cfg = json.load(f)
        assert cfg["model_name"] == "my_model"
        assert cfg["model_type"] == "gemma4"

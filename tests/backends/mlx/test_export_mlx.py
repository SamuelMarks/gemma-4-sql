"""Tests for MLX export."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import gemma_4_sql.backends.mlx.export as export_module
from gemma_4_sql.exceptions import DependencyMissingError, ExportError
from tests.conftest import requires_mlx

pytestmark = requires_mlx


def test_export_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify MLX export_model writes safetensors and config.json without sys.modules deletion."""
    mock_mx = MagicMock()
    monkeypatch.setattr(export_module, "mx", mock_mx)

    export_dir = tmp_path / "export"

    mock_load_fn = MagicMock()
    mock_model = MagicMock()
    mock_model.parameters.return_value = {"w": 1}.items()
    mock_model.config = {"k": "v"}
    mock_load_fn.return_value = (mock_model, "tok")
    monkeypatch.setattr(export_module, "mlx_load", mock_load_fn)

    res = export_module.export_model("model", str(export_dir))

    assert res["backend"] == "mlx"
    assert res["status"] == "exported_with_safetensors"
    assert res["format"] == "safetensors"
    assert (export_dir / "model.safetensors").parent.exists()
    mock_mx.save_safetensors.assert_called_once()

    config_path = export_dir / "config.json"
    assert config_path.exists()
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
        assert cfg == {"k": "v"}

    mock_model.config = MagicMock()
    mock_model.config.__dict__ = {"k2": "v2"}
    export_module.export_model("model", str(export_dir))
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
        assert cfg == {"k2": "v2"}

    del mock_model.config
    export_module.export_model("model", str(export_dir))
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
        assert cfg == {"model_type": "gemma4", "model_name": "model"}

    monkeypatch.setattr(export_module, "mlx_load", None)
    monkeypatch.setattr(export_module, "optional_attr", lambda *_a, **_k: None)
    with pytest.raises(ExportError, match="Failed to load MLX model"):
        export_module.export_model("model", str(export_dir))

    monkeypatch.setattr(export_module, "mx", None)
    with pytest.raises(DependencyMissingError, match="MLX is not installed"):
        export_module.export_model("model", str(export_dir))

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
    """Verify MLX export_model writes real safetensors with mlx.core.save_safetensors and config.json."""
    import mlx.core as mx

    export_dir = tmp_path / "export"
    expected_weight = mx.array([1.25, 2.5, 3.75], dtype=mx.float32)

    mock_load_fn = MagicMock()
    mock_model = MagicMock()
    mock_model.parameters.return_value = {"w": expected_weight}.items()
    mock_model.config = {"k": "v"}
    mock_load_fn.return_value = (mock_model, "tok")
    monkeypatch.setattr(export_module, "mlx_load", mock_load_fn)

    res = export_module.export_model("model", str(export_dir))

    assert res["backend"] == "mlx"
    assert res["status"] == "exported_with_safetensors"
    assert res["format"] == "safetensors"
    sf_file = export_dir / "model.safetensors"
    assert sf_file.is_file()
    loaded_tensors = mx.load(str(sf_file))
    assert isinstance(loaded_tensors, dict)
    assert bool(mx.allclose(loaded_tensors["w"], expected_weight))

    config_path = export_dir / "config.json"
    assert config_path.exists()
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
        assert cfg == {"k": "v"}

    from dataclasses import dataclass

    @dataclass
    class _DataclassConfig:
        """Dataclass config for MLX export testing."""

        k2: str = "v2"

    mock_model.config = _DataclassConfig()
    export_module.export_model("model", str(export_dir))
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
        assert cfg == {"k2": "v2"}

    del mock_model.config
    export_module.export_model("model", str(export_dir))
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
        assert cfg == {"model_type": "gemma4", "model_name": "model"}

    def _none_attr(*_a: str, **_k: str) -> None:
        """Return None for optional_attr."""
        return

    monkeypatch.setattr(export_module, "mlx_load", None)
    monkeypatch.setattr(export_module, "optional_attr", _none_attr)
    with pytest.raises(ExportError, match="Failed to load MLX model"):
        export_module.export_model("model", str(export_dir))

    monkeypatch.setattr(export_module, "mx", None)
    with pytest.raises(DependencyMissingError, match="MLX is not installed"):
        export_module.export_model("model", str(export_dir))

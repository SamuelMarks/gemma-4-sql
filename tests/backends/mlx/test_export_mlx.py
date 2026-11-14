"""Tests for MLX export without MagicMock."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

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

    @dataclass
    class _DataclassConfig:
        """Dataclass config for MLX export testing."""

        k2: str = "v2"

    class _ModelWithConfig:
        """Typed MLX model double exposing `.parameters()` and `.config`."""

        def __init__(self, cfg: dict[str, str] | _DataclassConfig | type[_DataclassConfig]) -> None:
            """Store config."""
            self.config = cfg

        def parameters(self) -> dict[str, mx.array]:
            """Return parameter dictionary."""
            return {"w": expected_weight}

    class _ModelWithoutConfig:
        """Typed MLX model double exposing `.parameters()` without `.config`."""

        def parameters(self) -> dict[str, mx.array]:
            """Return parameter dictionary."""
            return {"w": expected_weight}

    current_model: _ModelWithConfig | _ModelWithoutConfig = _ModelWithConfig({"k": "v"})

    def _fake_load(_name: str) -> tuple[_ModelWithConfig | _ModelWithoutConfig, str]:
        """Return `(current_model, 'tok')`."""
        return (current_model, "tok")

    monkeypatch.setattr(export_module, "mlx_load", _fake_load)

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

    current_model = _ModelWithConfig(_DataclassConfig())
    export_module.export_model("model", str(export_dir))
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
        assert cfg == {"k2": "v2"}

    current_model = _ModelWithConfig(_DataclassConfig)
    export_module.export_model("model", str(export_dir))
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
        assert cfg == {"model_type": "gemma4", "model_name": "model"}

    current_model = _ModelWithoutConfig()
    export_module.export_model("model", str(export_dir))
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
        assert cfg == {"model_type": "gemma4", "model_name": "model"}

    def _none_attr(*_a: str | type) -> None:
        """Return None for import_optional_symbol."""

    monkeypatch.setattr(export_module, "mlx_load", None)
    monkeypatch.setattr(export_module, "import_optional_symbol", _none_attr)
    with pytest.raises(ExportError, match="Failed to load MLX model"):
        export_module.export_model("model", str(export_dir))

    monkeypatch.setattr(export_module, "mx", None)
    with pytest.raises(DependencyMissingError, match="MLX is not installed"):
        export_module.export_model("model", str(export_dir))

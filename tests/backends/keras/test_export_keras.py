"""Unit and behavioral tests for Keras model export."""

from __future__ import annotations

import pathlib
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import requires_keras

pytestmark = requires_keras

import gemma_4_sql.backends.keras.export as mod
from gemma_4_sql.exceptions import DependencyMissingError, ExportError


def test_keras_export_missing_dependency(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """Verify `export_model` raises `DependencyMissingError` when `keras` is `None`."""
    monkeypatch.setattr(mod, "keras", None)
    with pytest.raises(DependencyMissingError, match="Keras dependencies are missing for export"):
        mod.export_model("my_model", str(tmp_path))


def test_export_model_keras_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `export_model` saves the Keras model and raises `ExportError` on load failure."""
    mock_cls = MagicMock()
    monkeypatch.setattr(mod, "keras", MagicMock())
    monkeypatch.setattr(mod, "GemmaCausalLM", mock_cls)

    with patch.object(pathlib.Path, "mkdir"):
        res = mod.export_model("my_model", "path")
        assert res["status"] == "exported_with_keras"
        assert res["format"] == "keras_v3/keras_tensor"
        mock_cls.from_preset.assert_called_with("my_model")
        mock_cls.from_preset.return_value.save.assert_called_once()

        # ValueError during preset loading raises ExportError (a ValueError & RuntimeError subclass)
        mock_cls.from_preset.side_effect = ValueError("fake error")
        with pytest.raises(ExportError, match="Failed to load model my_model"):
            mod.export_model("my_model", "path")

        # Missing GemmaCausalLM class raises ExportError
        monkeypatch.setattr(mod, "GemmaCausalLM", None)
        monkeypatch.setattr(mod, "optional_attr", lambda *_a, **_k: None)  # pyright: ignore[reportUnknownLambdaType]
        with pytest.raises(ExportError, match="Failed to load model my_model"):
            mod.export_model("my_model", "path")

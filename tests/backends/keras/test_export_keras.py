"""Tests for Keras export."""

import builtins
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.backends.keras.export import export_model
from gemma_4_sql.exceptions import DependencyMissingError

original_import = builtins.__import__


def _mock_import(name, *args, **kwargs):
    """Docstring for _mock_import."""
    if name == "keras_nlp.models":
        raise ImportError
    return original_import(name, *args, **kwargs)


def test_export_model_missing_keras():
    """Docstring for test_export_model_missing_keras."""
    with patch("gemma_4_sql.backends.keras.export.keras", None), pytest.raises(DependencyMissingError, match="Keras dependencies are missing for export"):
        export_model("model", "/tmp/export")


def test_export_model_import_error():
    """Docstring for test_export_model_import_error."""
    mock_keras = MagicMock()
    with patch("gemma_4_sql.backends.keras.export.keras", mock_keras), patch("builtins.__import__", side_effect=_mock_import), pytest.raises(ValueError, match="Failed to load model"):
        export_model("model", "/tmp/export")


def test_export_model_success():
    """Docstring for test_export_model_success."""
    mock_keras = MagicMock()
    mock_cls = MagicMock()
    mock_model = MagicMock()
    mock_cls.GemmaCausalLM.from_preset.return_value = mock_model

    def _mock_import_success(name, *args, **kwargs):
        """Docstring for _mock_import_success."""
        if name == "keras_nlp.models":
            return mock_cls
        return original_import(name, *args, **kwargs)

    with patch("gemma_4_sql.backends.keras.export.keras", mock_keras), patch("builtins.__import__", side_effect=_mock_import_success):
        res = export_model("model", "/tmp/export")
        assert res["status"] == "exported_with_keras"
        assert res["export_path"] == "/tmp/export"
        assert res["file_path"] == "/tmp/export/model.keras"
        mock_model.save.assert_called_once_with(Path("/tmp/export/model.keras"))

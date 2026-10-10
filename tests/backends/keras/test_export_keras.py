"""Module docstring."""

import builtins
import importlib

import pytest

import gemma_4_sql.backends.keras.export as mod


def test_keras_export_import_error():
    """Docstring for test_keras_export_import_error."""
    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        """Docstring for mock_import."""
        if name == "keras_nlp" or name == "keras":
            raise ImportError("mock")
        return orig_import(name, *args, **kwargs)

    builtins.__import__ = mock_import
    try:
        importlib.reload(mod)
        assert mod.keras is None
    finally:
        builtins.__import__ = orig_import
        importlib.reload(mod)


def test_export_model_keras_success():
    """Docstring for test_export_model_keras_success."""
    import pathlib
    from unittest.mock import MagicMock, patch

    import gemma_4_sql.backends.keras.export as mod

    with patch.object(mod, "keras", MagicMock()):
        with patch.object(pathlib.Path, "mkdir"):
            with patch("builtins.__import__") as mock_import:
                mock_cls = MagicMock()
                mock_import.return_value.GemmaCausalLM = mock_cls

                # Test success
                res = mod.export_model("my_model", "path")
                assert res["status"] == "exported_with_keras"
                mock_cls.from_preset.assert_called_with("my_model")
                mock_cls.from_preset.return_value.save.assert_called_once()

                # Test ValueError branch
                mock_cls.from_preset.side_effect = ValueError("fake error")
                with pytest.raises(ValueError, match="Failed to load model my_model"):
                    mod.export_model("my_model", "path")

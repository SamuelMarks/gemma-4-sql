"""Module docstring."""

import builtins
import importlib
from unittest.mock import MagicMock, patch

import pytest

import gemma_4_sql.backends.keras.serve as mod


def test_keras_serve_import_error():
    """Docstring for test_keras_serve_import_error."""
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


def test_keras_serve_branch():
    """Docstring for test_keras_serve_branch."""
    with patch.object(mod, "keras", MagicMock()):
        with patch("gemma_4_sql.backends.keras.serve.create_common_app") as mock_create:
            mod.create_app("test_model")

            # The inner functions _startup, _generate, _batch_generate are passed to create_common_app
            kwargs = mock_create.call_args.kwargs
            _startup = kwargs["startup_callback"]
            _generate = kwargs["generate_logic"]
            _batch_generate = kwargs["batch_generate_logic"]

            # Test _startup error branch
            with patch("builtins.__import__") as mock_import:
                mock_import.return_value.GemmaCausalLM.from_preset.side_effect = ValueError("mock err")
                _startup()

            # Test _generate error branch
            mod.loaded_model = MagicMock()
            mod.loaded_model.generate.side_effect = ValueError("mock err")
            with pytest.raises(Exception):
                _generate("test")

            # Test _batch_generate error branch (fallback to _generate)
            mod.loaded_model.generate.side_effect = ValueError("mock err")
            # We must mock _generate because it will fallback to it
            # But _generate will raise ValueError since we just set side_effect
            with pytest.raises(Exception):
                _batch_generate(["test1", "test2"])

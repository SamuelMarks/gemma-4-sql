"""Module docstring."""

import builtins
import importlib
from unittest.mock import MagicMock, patch

import pytest

import gemma_4_sql.backends.maxtext.serve as mod


def test_maxtext_serve_import_error():
    """Docstring for test_maxtext_serve_import_error."""
    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        """Docstring for mock_import."""
        if name == "jax" or name == "maxtext":
            raise ImportError("mock")
        return orig_import(name, *args, **kwargs)

    builtins.__import__ = mock_import
    try:
        importlib.reload(mod)
        assert mod.jax is None
    finally:
        builtins.__import__ = orig_import
        importlib.reload(mod)


def test_maxtext_serve_branches():
    """Docstring for test_maxtext_serve_branches."""
    from gemma_4_sql.exceptions import InferenceError

    with patch.object(mod, "jax", MagicMock()):
        with patch("gemma_4_sql.backends.maxtext.serve.create_common_app") as mock_create:
            mod._create_app("test_model")

            kwargs = mock_create.call_args.kwargs
            _generate = kwargs["generate_logic"]

            with patch("gemma_4_sql.backends.maxtext.inference.generate_sql") as mock_gen:
                mock_gen.return_value = {"sql": "SELECT 1"}
                assert _generate("p") == "SELECT 1"

                mock_gen.return_value = {"sql": ""}
                with pytest.raises(InferenceError):
                    _generate("p")

                mock_gen.side_effect = ValueError("mock err")
                with pytest.raises(InferenceError):
                    _generate("p")

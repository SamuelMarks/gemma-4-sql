"""Module docstring."""

import builtins
import importlib
from unittest.mock import MagicMock, patch

import gemma_4_sql.backends.maxtext.train as mod


def test_maxtext_train_import_error():
    """Docstring for test_maxtext_train_import_error."""
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


def test_maxtext_train_branches():
    """Docstring for test_maxtext_train_branches."""
    from gemma_4_sql.type_hints import TrainingConfig

    config = TrainingConfig(model_name="model", dataset="dataset", epochs=1, batch_size=1)

    with patch.object(mod, "jax", MagicMock()):
        with patch.object(mod, "jnp", MagicMock()):
            with patch.object(mod, "optax", MagicMock()):
                with patch.object(mod, "Gemma4Model", MagicMock()):
                    with patch("gemma_4_sql.backends.maxtext.train._execute_train", return_value=("completed", 1.0)):
                        res = mod.train_model(config)
                        assert res["status"] == "completed"

                    with patch("gemma_4_sql.backends.maxtext.train._execute_train", side_effect=ValueError("mock err")):
                        res = mod.train_model(config)
                        assert "failed:" in res["status"]

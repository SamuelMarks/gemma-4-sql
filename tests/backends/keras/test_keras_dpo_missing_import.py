"""Tests for Keras DPO missing imports."""

import builtins
import importlib


def test_keras_dpo_missing_imports(monkeypatch):
    """Test keras DPO missing imports."""
    from gemma_4_sql.backends.keras import dpo

    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        """Docstring for mock_import."""
        if name == "keras" or name == "tensorflow":
            raise ImportError()
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    importlib.reload(dpo)

    assert dpo.keras is None
    assert dpo.tf is None

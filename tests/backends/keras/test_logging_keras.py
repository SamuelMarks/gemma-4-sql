"""Module docstring."""

from unittest.mock import MagicMock

import pytest


def test_log_metrics(monkeypatch):
    """Docstring for test_log_metrics."""
    import gemma_4_sql.backends.keras.logging as log_keras
    from gemma_4_sql.exceptions import DependencyMissingError

    monkeypatch.setattr(log_keras, "tf", None)
    with pytest.raises(DependencyMissingError):
        log_keras.log_metrics({}, 1)

    mock_tf = MagicMock()
    monkeypatch.setattr(log_keras, "tf", mock_tf)

    class MockWriter:
        """Docstring for MockWriter."""

        def __init__(self, *args, **kwargs):
            """Docstring for __init__."""

        def as_default(self):
            """Docstring for as_default."""
            return MagicMock()

        def close(self):
            """Docstring for close."""

    mock_tf.summary.create_file_writer.return_value = MockWriter()
    res = log_keras.log_metrics({"a": 1.0}, 1)
    assert res["status"] == "success"

    del mock_tf.summary
    res2 = log_keras.log_metrics({"a": 1.0}, 1)
    assert res2["status"] == "missing_summary_attr"


def test_module_load_import_error():
    """Docstring for test_module_load_import_error."""
    import gemma_4_sql.backends.keras.logging as q

    with open(q.__file__) as f:
        code = f.read()

    import builtins

    orig_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        """Docstring for mock_import."""
        if name == "tensorflow":
            raise ImportError("simulated missing import")
        return orig_import(name, *args, **kwargs)

    namespace = {"__name__": "mock_logging", "__builtins__": dict(builtins.__dict__)}
    namespace["__builtins__"]["__import__"] = mock_import

    exec(code, namespace)  # noqa: S102

    assert namespace.get("tf") is None


def test_module_load_import_error_coverage(monkeypatch):
    """Docstring for test_module_load_import_error_coverage."""
    import importlib
    import sys

    import gemma_4_sql.backends.keras.logging as log_keras

    monkeypatch.setitem(sys.modules, "tensorflow", None)
    importlib.reload(log_keras)
    assert log_keras.tf is None
    monkeypatch.undo()
    importlib.reload(log_keras)

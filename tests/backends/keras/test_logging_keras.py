"""Unit and behavioral tests for logging keras."""

from unittest.mock import MagicMock

from tests.conftest import requires_keras

pytestmark = requires_keras

import pytest


def test_log_metrics(monkeypatch):
    """Execute and verify `test_log_metrics` behavior and invariants."""
    import gemma_4_sql.backends.keras.logging as log_keras
    from gemma_4_sql.exceptions import DependencyMissingError

    monkeypatch.setattr(log_keras, "tf", None)
    with pytest.raises(DependencyMissingError):
        log_keras.log_metrics({}, 1)

    mock_tf = MagicMock()
    monkeypatch.setattr(log_keras, "tf", mock_tf)

    class MockWriter:
        """Execute and verify `MockWriter` behavior and invariants."""

        def __init__(self, *args, **kwargs):
            """Execute and verify `__init__` behavior and invariants."""

        def as_default(self):
            """Execute and verify `as_default` behavior and invariants."""
            return MagicMock()

        def close(self):
            """Execute and verify `close` behavior and invariants."""

    mock_tf.summary.create_file_writer.return_value = MockWriter()
    res = log_keras.log_metrics({"a": 1.0}, 1)
    assert res["status"] == "success"

    del mock_tf.summary
    res2 = log_keras.log_metrics({"a": 1.0}, 1)
    assert res2["status"] == "missing_summary_attr"


def test_module_load_import_error():
    """Execute and verify `test_module_load_import_error` behavior and invariants."""
    import importlib
    import sys
    from unittest.mock import patch

    import gemma_4_sql.backends.keras.logging as q

    with patch.dict(sys.modules, {"tensorflow": None, "tensorboard": None}):
        importlib.reload(q)
        assert getattr(q, "tf", None) is None

    importlib.reload(q)


def test_module_load_import_error_coverage(monkeypatch):
    """Execute and verify `test_module_load_import_error_coverage` behavior and invariants."""
    import importlib
    import sys

    import gemma_4_sql.backends.keras.logging as log_keras

    monkeypatch.setitem(sys.modules, "tensorflow", None)
    importlib.reload(log_keras)
    assert log_keras.tf is None
    monkeypatch.undo()
    importlib.reload(log_keras)

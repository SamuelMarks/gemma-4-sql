"""Unit and behavioral tests for logging keras."""

from typing import Any
from unittest.mock import MagicMock

from tests.conftest import requires_keras

pytestmark = requires_keras

import pytest


def test_log_metrics(monkeypatch: pytest.MonkeyPatch):
    """Verify log metrics behavior."""
    import gemma_4_sql.backends.keras.logging as log_keras
    from gemma_4_sql.exceptions import DependencyMissingError

    monkeypatch.setattr(log_keras, "tf", None)
    with pytest.raises(DependencyMissingError):
        log_keras.log_metrics({}, 1)

    mock_tf = MagicMock()
    monkeypatch.setattr(log_keras, "tf", mock_tf)

    class MockWriter:
        """Test stub for `MockWriter`."""

        def __init__(self, *args: Any, **kwargs: Any):
            """Initialize test stub instance."""

        def as_default(self):
            """Provide as default test helper."""
            return MagicMock()

        def close(self):
            """Provide close test helper."""

    mock_tf.summary.create_file_writer.return_value = MockWriter()
    res = log_keras.log_metrics({"a": 1.0}, 1)
    assert res["status"] == "success"

    del mock_tf.summary
    res2 = log_keras.log_metrics({"a": 1.0}, 1)
    assert res2["status"] == "missing_summary_attr"


def test_module_load_import_error():
    """Verify module load import error behavior."""
    import importlib
    import sys
    from unittest.mock import patch

    import gemma_4_sql.backends.keras.logging as q

    with patch.dict(sys.modules, {"tensorflow": None, "tensorboard": None}):
        importlib.reload(q)
        assert getattr(q, "tf", None) is None

    importlib.reload(q)


def test_module_load_import_error_coverage(monkeypatch: pytest.MonkeyPatch):
    """Verify module load import error coverage behavior."""
    import importlib
    import sys

    import gemma_4_sql.backends.keras.logging as log_keras

    monkeypatch.setitem(sys.modules, "tensorflow", None)
    importlib.reload(log_keras)
    assert log_keras.tf is None
    monkeypatch.undo()
    importlib.reload(log_keras)

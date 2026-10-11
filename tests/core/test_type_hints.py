"""Tests for type hints module."""

import sys
import pytest
from unittest.mock import patch


def test_type_hints_protocols():
    """Test TensorType and ModelType protocols."""
    from gemma_4_sql.type_hints import TensorType, ModelType

    class DummyTensor:
        shape = (1, 2)
        dtype = "float32"

        def tolist(self):
            return [1, 2]

        def to(self, device):
            return self

        def item(self):
            return 1.0

        def cpu(self):
            return self

        def numpy(self):
            return [1.0, 2.0]

        def view(self, *a, **k):
            return self

        def reshape(self, *a, **k):
            return self

    t = DummyTensor()
    assert hasattr(t, "shape")


def test_numpy_fallback(monkeypatch):
    """Test fallback when numpy is missing."""
    import gemma_4_sql.type_hints as th
    import importlib

    with patch.dict(sys.modules, {"numpy": None}):
        importlib.reload(th)
        assert th.ndarray is object
    importlib.reload(th)

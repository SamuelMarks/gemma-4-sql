"""Tests for Keras export."""

import os
import sys
import tempfile

import pytest


class MockModel:
    def save(self, path):
        with open(path, "w") as f:
            f.write("saved")


class MockKeras:
    class Input:
        def __init__(self, shape):
            pass

    class layers:
        class Dense:
            def __init__(self, units):
                pass

            def __call__(self, inputs):
                return "out"

    @classmethod
    def Model(cls, inputs, outputs):
        return MockModel()


@pytest.fixture
def mock_keras_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "keras", MockKeras())
    import gemma_4_sql.backends.keras.export as exp

    monkeypatch.setattr(exp, "keras", MockKeras())


def test_export_keras_mocked() -> None:
    import gemma_4_sql.backends.keras.export as exp
    from gemma_4_sql.backends.keras.export import export_model

    orig_keras = exp.keras
    exp.keras = None
    with tempfile.TemporaryDirectory() as tmpdir:
        res = export_model("mod", tmpdir)
        assert res["status"] == "mock_exported"
        assert os.path.exists(res["file_path"])
    exp.keras = orig_keras


def test_export_keras_real(mock_keras_env: None) -> None:
    from gemma_4_sql.backends.keras.export import export_model

    with tempfile.TemporaryDirectory() as tmpdir:
        res = export_model("mod", tmpdir)
        assert res["status"] == "exported_with_keras"
        assert os.path.exists(res["file_path"])

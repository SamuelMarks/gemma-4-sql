"""Tests for PyTorch export."""

import os
import tempfile

import pytest


class MockTorch:
    @staticmethod
    def zeros(shape):
        return "zeros"


def mock_save_file(tensors, path):
    with open(path, "w") as f:
        f.write("saved")


@pytest.fixture
def mock_pytorch_env(monkeypatch: pytest.MonkeyPatch) -> None:
    import gemma_4_sql.backends.pytorch.export as exp

    monkeypatch.setattr(exp, "torch", MockTorch())
    monkeypatch.setattr(exp, "save_file", mock_save_file)


def test_export_pytorch_mocked() -> None:
    import gemma_4_sql.backends.pytorch.export as exp
    from gemma_4_sql.backends.pytorch.export import export_model

    orig_torch = exp.torch
    exp.torch = None
    with tempfile.TemporaryDirectory() as tmpdir:
        res = export_model("mod", tmpdir)
        assert res["status"] == "mock_exported"
        assert os.path.exists(res["file_path"])
    exp.torch = orig_torch


def test_export_pytorch_real(mock_pytorch_env: None) -> None:
    from gemma_4_sql.backends.pytorch.export import export_model

    with tempfile.TemporaryDirectory() as tmpdir:
        res = export_model("mod", tmpdir)
        assert res["status"] == "exported_with_safetensors"
        assert os.path.exists(res["file_path"])

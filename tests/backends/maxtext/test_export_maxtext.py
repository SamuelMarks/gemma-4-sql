"""Tests for MaxText export."""

import os
import sys
import tempfile

import pytest


class MockJNP:
    def zeros(self, shape):
        return "zeros"


class MockManager:
    def __init__(self, path, checkpointer, options):
        self.path = path

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def save(self, step, tree):
        os.makedirs(self.path, exist_ok=True)
        with open(os.path.join(self.path, "saved"), "w") as f:
            f.write("saved")


class MockOCP:
    @staticmethod
    def PyTreeCheckpointer():
        return "checkpointer"

    @staticmethod
    def CheckpointManagerOptions(max_to_keep):
        return "options"

    CheckpointManager = MockManager


@pytest.fixture
def mock_maxtext_env(monkeypatch: pytest.MonkeyPatch) -> None:
    import gemma_4_sql.backends.maxtext.export as exp

    monkeypatch.setattr(exp, "jnp", MockJNP())
    monkeypatch.setattr(exp, "ocp", MockOCP())
    monkeypatch.setattr(exp, "jax", sys.modules["jax"])


def test_export_maxtext_mocked() -> None:
    import gemma_4_sql.backends.maxtext.export as exp
    from gemma_4_sql.backends.maxtext.export import export_model

    orig_jnp = exp.jnp
    exp.jnp = None
    with tempfile.TemporaryDirectory() as tmpdir:
        res = export_model("mod", tmpdir)
        assert res["status"] == "mock_exported"
        assert os.path.exists(res["file_path"])
    exp.jnp = orig_jnp


def test_export_maxtext_real(mock_maxtext_env: None) -> None:
    from gemma_4_sql.backends.maxtext.export import export_model

    with tempfile.TemporaryDirectory() as tmpdir:
        res = export_model("mod", tmpdir)
        assert res["status"] == "exported_with_maxtext_orbax"
        assert os.path.exists(res["file_path"])

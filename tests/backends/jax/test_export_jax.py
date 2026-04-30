"""Tests for JAX export."""

import os
import sys
import tempfile

import pytest


class MockJNP:
    def zeros(self, shape):
        return "zeros"


class MockCheckpointer:
    def save(self, path, tree):
        with open(path, "w") as f:
            f.write("saved")


class MockOCP:
    @staticmethod
    def PyTreeCheckpointer():
        return MockCheckpointer()


@pytest.fixture
def mock_jax_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "jax", type("jax", (), {"numpy": MockJNP()})())
    monkeypatch.setitem(sys.modules, "jax.numpy", MockJNP())
    monkeypatch.setitem(
        sys.modules, "orbax", type("orbax", (), {"checkpoint": MockOCP()})()
    )
    monkeypatch.setitem(sys.modules, "orbax.checkpoint", MockOCP())
    import gemma_4_sql.backends.jax.export as exp

    monkeypatch.setattr(exp, "jnp", MockJNP())
    monkeypatch.setattr(exp, "ocp", MockOCP())
    monkeypatch.setattr(exp, "jax", sys.modules["jax"])


def test_export_jax_mocked() -> None:
    import gemma_4_sql.backends.jax.export as exp
    from gemma_4_sql.backends.jax.export import export_model

    orig_jnp = exp.jnp
    exp.jnp = None
    with tempfile.TemporaryDirectory() as tmpdir:
        res = export_model("mod", tmpdir)
        assert res["status"] == "mock_exported"
        assert os.path.exists(res["file_path"])
    exp.jnp = orig_jnp


def test_export_jax_real(mock_jax_env: None) -> None:
    from gemma_4_sql.backends.jax.export import export_model

    with tempfile.TemporaryDirectory() as tmpdir:
        res = export_model("mod", tmpdir)
        assert res["status"] == "exported_with_orbax"
        assert os.path.exists(res["file_path"])

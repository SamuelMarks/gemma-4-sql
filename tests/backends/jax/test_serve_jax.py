"""Provide module docstring."""

import sys
from unittest import mock

import pytest

import gemma_4_sql.backends.jax.serve as srv
from gemma_4_sql.exceptions import DependencyMissingError


def test_serve_model_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    """Initialize function test_serve_model_jax."""
    monkeypatch.setattr(srv, "jax", object())

    def mock_serve_model_wrapper(backend_name, model_name, port, max_batch_size, missing_deps, missing_status, app_factory):
        # Trigger the factory to test its internal logic
        app = app_factory()

        # Test startup callback
        startup = app.get("startup_callback")

        # Mock generate_sql for startup warmup
        with mock.patch("gemma_4_sql.backends.jax.inference.generate_sql") as mock_gen:
            startup()
            mock_gen.assert_called_once_with(model_name=model_name, prompt="SELECT 1")

        # Test startup warmup with error
        with mock.patch("gemma_4_sql.backends.jax.inference.generate_sql", side_effect=RuntimeError("Warmup fail")):
            startup()  # Should not raise

        # Test generate_logic
        generate = app.get("generate_logic")
        with mock.patch("gemma_4_sql.backends.jax.inference.generate_sql", return_value={"sql": "SELECT 42"}) as mock_gen:
            res = generate("test prompt")
            assert res == "SELECT 42"

        # Test generate_logic without sql key
        with mock.patch("gemma_4_sql.backends.jax.inference.generate_sql", return_value={"other": "data"}):
            res = generate("test prompt")
            assert "SELECT * FROM generated WHERE prompt='test prompt'" in res

        # Test generate_logic error
        with mock.patch("gemma_4_sql.backends.jax.inference.generate_sql", side_effect=RuntimeError("Gen fail")):
            res = generate("test prompt")
            assert "SELECT * FROM generated WHERE prompt='test prompt'" in res

        # Test batch_generate_logic
        batch_generate = app.get("batch_generate_logic")
        with mock.patch("gemma_4_sql.backends.jax.inference.generate_sql", return_value={"sql": "SELECT 42"}):
            res_batch = batch_generate(["prompt 1", "prompt 2"])
            assert res_batch == ["SELECT 42", "SELECT 42"]

        return {"backend": backend_name, "model": model_name, "port": port, "max_batch_size": max_batch_size, "mode": "continuous_batching", "status": "running_jax_serve"}

    def mock_create_common_app(**kwargs):
        return kwargs

    monkeypatch.setattr(srv, "serve_model_wrapper", mock_serve_model_wrapper)
    monkeypatch.setattr(srv, "create_common_app", mock_create_common_app)

    res = srv.serve_model("foo", port=8000, max_batch_size=16)
    assert res["backend"] == "jax"
    assert res["model"] == "foo"
    assert res["port"] == 8000
    assert res["max_batch_size"] == 16
    assert res["mode"] == "continuous_batching"
    assert res["status"] == "running_jax_serve"


def test_serve_model_jax_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Initialize function test_serve_model_jax_missing."""
    monkeypatch.setattr(srv, "jax", None)
    with pytest.raises(DependencyMissingError, match=r"JAX dependencies are missing for serve\."):
        srv.serve_model("foo")


def test_serve_model_jax_other_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test when status is not running_jax_serve."""
    monkeypatch.setattr(srv, "jax", object())

    def mock_serve_model_wrapper(**kwargs):
        return {"status": "mocked_missing_jax", "backend": "jax"}

    monkeypatch.setattr(srv, "serve_model_wrapper", mock_serve_model_wrapper)
    res = srv.serve_model("foo")
    assert res["status"] == "mocked_missing_jax"


def test_serve_imports_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    import importlib

    orig_jax = srv.jax

    with mock.patch.dict(sys.modules, {"jax": None}):
        importlib.reload(srv)
        assert srv.jax is None

    srv.jax = orig_jax

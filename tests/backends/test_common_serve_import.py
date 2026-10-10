"""Module docstring."""

import importlib
import sys

import pytest

import gemma_4_sql.backends.common_serve as mod


def test_fastapi_missing(monkeypatch):
    """Docstring for test_fastapi_missing."""
    monkeypatch.setitem(sys.modules, "fastapi", None)
    monkeypatch.setitem(sys.modules, "uvicorn", None)

    importlib.reload(mod)
    assert mod.FastAPI is None

    from gemma_4_sql.exceptions import DependencyMissingError

    with pytest.raises(DependencyMissingError):
        mod.serve_model_wrapper(backend_name="pytorch", model_name="test", max_batch_size=32, missing_deps=None, missing_status=None, app_factory=lambda: None, port=8000)

    # Restore
    monkeypatch.delitem(sys.modules, "fastapi", raising=False)
    monkeypatch.delitem(sys.modules, "uvicorn", raising=False)
    # the sys.modules will be restored by monkeypatch at the end of the test.
    importlib.reload(mod)


def test_serve_model_wrapper_run_server():
    """Docstring for test_serve_model_wrapper_run_server."""
    import pytest

    import gemma_4_sql.backends.common_serve as mod

    # ensure it's reloaded with correct deps first
    importlib.reload(mod)

    with pytest.MonkeyPatch.context() as mp:
        called = []

        class MockUvicorn:
            """Docstring for MockUvicorn."""

            def run(self, *args, **kwargs):
                """Docstring for run."""
                called.append(True)

        mp.setattr(mod, "uvicorn", MockUvicorn())

        mod.serve_model_wrapper(backend_name="pytorch", model_name="test", max_batch_size=32, missing_deps=None, missing_status=None, app_factory=lambda: "fake_app", run_server=True, port=8000)
        assert len(called) == 1

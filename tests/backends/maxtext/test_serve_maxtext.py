from unittest.mock import MagicMock

import pytest

import gemma_4_sql.backends.maxtext.serve as serve_module
from gemma_4_sql.exceptions import DependencyMissingError, InferenceError


@pytest.fixture
def mock_jax_deps(monkeypatch):
    mock_jax = MagicMock()
    mock_gemma4 = MagicMock()
    mock_jax.distributed.initialize = MagicMock()

    monkeypatch.setattr(serve_module, "jax", mock_jax)
    monkeypatch.setattr(serve_module, "gemma4", mock_gemma4)

    # Mock create_common_app and serve_model_wrapper
    mock_create_common = MagicMock(return_value="app")
    mock_serve_wrapper = MagicMock(return_value={"status": "serving"})

    monkeypatch.setattr(serve_module, "create_common_app", mock_create_common)
    monkeypatch.setattr(serve_module, "serve_model_wrapper", mock_serve_wrapper)

    return {
        "jax": mock_jax,
        "gemma4": mock_gemma4,
        "create_common_app": mock_create_common,
        "serve_model_wrapper": mock_serve_wrapper,
    }


def test_serve_model(mock_jax_deps):
    res = serve_module.serve_model("dummy_model", port=8000, max_batch_size=32)
    assert res["status"] == "serving"
    mock_jax_deps["serve_model_wrapper"].assert_called_once()

    # Get the app_factory to test _create_app
    app_factory = mock_jax_deps["serve_model_wrapper"].call_args[1]["app_factory"]
    app = app_factory()
    assert app == "app"

    # Get callbacks
    startup_cb = mock_jax_deps["create_common_app"].call_args[1]["startup_callback"]
    generate_logic = mock_jax_deps["create_common_app"].call_args[1]["generate_logic"]

    # Test startup
    startup_cb()
    mock_jax_deps["jax"].distributed.initialize.assert_called_once()

    # Test generate_logic (it will fail to import missing module or raise InferenceError)
    with pytest.raises(Exception):
        generate_logic("prompt")


def test_serve_model_errors(mock_jax_deps, monkeypatch):
    monkeypatch.setattr(serve_module, "jax", None)
    with pytest.raises(DependencyMissingError):
        serve_module.serve_model("dummy_model")


def test_startup_callback_error(mock_jax_deps):
    mock_jax_deps["jax"].distributed.initialize.side_effect = RuntimeError("init fail")
    serve_module._create_app("dummy_model")
    startup_cb = mock_jax_deps["create_common_app"].call_args[1]["startup_callback"]
    startup_cb()  # Should catch the error and log it, not raise


def test_generate_logic(mock_jax_deps, monkeypatch):
    import sys

    mock_inference = MagicMock()
    mock_generate_sql = MagicMock()
    mock_inference.generate_sql = mock_generate_sql
    sys.modules["gemma_4_sql.backends.maxtext.inference"] = mock_inference

    serve_module._create_app("dummy_model")
    generate_logic = mock_jax_deps["create_common_app"].call_args[1]["generate_logic"]

    mock_generate_sql.return_value = {"sql": "SELECT 1"}
    assert generate_logic("prompt") == "SELECT 1"

    # empty sql
    mock_generate_sql.return_value = {"sql": ""}
    with pytest.raises(InferenceError, match="empty SQL"):
        generate_logic("prompt")

    # general exception
    mock_generate_sql.side_effect = ValueError("inference fail")
    with pytest.raises(InferenceError, match="generation failed"):
        generate_logic("prompt")

    # inference error
    mock_generate_sql.side_effect = InferenceError("inf error")
    with pytest.raises(InferenceError, match="inf error"):
        generate_logic("prompt")


def test_imports_except_blocks():
    import importlib
    import sys

    # Save original modules
    orig_jax = sys.modules.get("jax")
    orig_gemma = sys.modules.get("maxtext.models.gemma4")

    # Force ImportError
    sys.modules["jax"] = None
    sys.modules["maxtext.models.gemma4"] = None

    import gemma_4_sql.backends.maxtext.serve as sm

    importlib.reload(sm)

    assert sm.jax is None
    assert sm.gemma4 is None

    # Restore
    if orig_jax is not None:
        sys.modules["jax"] = orig_jax
    else:
        del sys.modules["jax"]

    if orig_gemma is not None:
        sys.modules["maxtext.models.gemma4"] = orig_gemma
    else:
        del sys.modules["maxtext.models.gemma4"]

    importlib.reload(sm)

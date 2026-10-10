"""Tests for PyTorch serve."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.exceptions import DependencyMissingError


def test_pytorch_serve_imports():
    """Test pytorch serve imports fallback."""
    import importlib

    with patch.dict(sys.modules, {"torch": None, "fastapi": None, "vllm": None}):
        import gemma_4_sql.backends.pytorch.serve as serve_module

        importlib.reload(serve_module)
        assert serve_module.torch is None
        assert serve_module.FastAPI is None
        assert serve_module.AsyncEngineArgs is None
        assert serve_module.AsyncLLMEngine is None

    with patch.dict(sys.modules, {"torch": MagicMock(), "fastapi": MagicMock(), "vllm": MagicMock()}):
        importlib.reload(serve_module)
    importlib.reload(serve_module)


@pytest.mark.asyncio
async def test_create_vllm_app(monkeypatch):
    """Test _create_vllm_app."""
    import gemma_4_sql.backends.pytorch.serve as serve_module

    mock_fastapi = MagicMock()
    mock_app = MagicMock()

    mock_fastapi.__call__ = MagicMock(return_value=mock_app)

    monkeypatch.setattr(serve_module, "FastAPI", mock_fastapi)
    monkeypatch.setattr(serve_module, "AsyncEngineArgs", MagicMock())
    monkeypatch.setattr(serve_module, "AsyncLLMEngine", MagicMock())
    monkeypatch.setattr(serve_module, "JSONResponse", MagicMock())

    app = serve_module._create_vllm_app("m", 1)
    assert app == mock_app

    # Check that route was registered
    mock_app.post.assert_called_with("/generate")
    generate_fn = mock_app.post.call_args_list[0][0][0]  # No wait, it's a decorator

    # We can inspect the decorator by capturing it
    def mock_decorator(path):
        """Docstring for mock_decorator."""

        def wrapper(func):
            """Docstring for wrapper."""
            mock_app.generate_func = func
            return func

        return wrapper

    mock_app.post.side_effect = mock_decorator
    serve_module._create_vllm_app("m", 1)
    generate_fn = mock_app.generate_func

    # Test generation execution
    mock_req = MagicMock()

    async def mock_json():
        """Docstring for mock_json."""
        return {"prompt": "p"}

    mock_req.json = mock_json

    async def mock_disconnected():
        """Docstring for mock_disconnected."""
        return False

    mock_req.is_disconnected = mock_disconnected

    mock_engine = MagicMock()
    serve_module.AsyncLLMEngine.from_engine_args.return_value = mock_engine

    class MockOutput:
        """Docstring for MockOutput."""

        class Out:
            """Docstring for Out."""

            text = "sql"

        outputs = [Out()]

    async def mock_generate_stream(*args, **kwargs):
        """Docstring for mock_generate_stream."""
        yield MockOutput()

    mock_engine.generate.return_value = mock_generate_stream()

    def mock_json_response(content):
        """Docstring for mock_json_response."""
        return {"response": content}

    serve_module.JSONResponse.__call__ = mock_json_response

    app = serve_module._create_vllm_app("m", 1)
    generate_fn = mock_app.generate_func

    res = await generate_fn(mock_req)
    assert res == {"response": {"sql": "sql"}}

    # Test disconnect
    async def mock_disconnected_true():
        """Docstring for mock_disconnected_true."""
        return True

    mock_req.is_disconnected = mock_disconnected_true

    async def mock_abort(*args):
        """Docstring for mock_abort."""

    mock_engine.abort = mock_abort

    mock_engine.generate.return_value = mock_generate_stream()

    res2 = await generate_fn(mock_req)
    assert res2 == {"response": {"error": "Client disconnected"}}

    # Test missing app or JSONResponse
    serve_module.JSONResponse = None
    app = serve_module._create_vllm_app("m", 1)
    assert app == mock_app


def test_create_native_app():
    """Test _create_native_app."""
    import gemma_4_sql.backends.pytorch.serve as serve_module

    with patch("gemma_4_sql.backends.pytorch.serve.create_common_app") as mock_create:
        serve_module._create_native_app("m", 1)
        mock_create.assert_called_once()

        generate_logic = mock_create.call_args[1]["generate_logic"]
        batch_generate_logic = mock_create.call_args[1]["batch_generate_logic"]

        with patch("gemma_4_sql.backends.pytorch.inference.generate_sql") as mock_gen:
            mock_gen.return_value = {"sql": "SELECT 1;"}
            assert generate_logic("p") == "SELECT 1;"

            mock_gen.return_value = {}
            assert "pytorch_native" in generate_logic("p")

            mock_gen.side_effect = RuntimeError("error")
            assert "pytorch_native" in generate_logic("p")

        with patch("gemma_4_sql.backends.pytorch.inference.generate_sql") as mock_gen:
            mock_gen.return_value = {"sql": "sql"}
            assert batch_generate_logic(["p1", "p2"]) == ["sql", "sql"]


def test_serve_model(monkeypatch):
    """Test serve_model."""
    import gemma_4_sql.backends.pytorch.serve as serve_module

    monkeypatch.setattr(serve_module, "AsyncEngineArgs", MagicMock())

    with patch("gemma_4_sql.backends.pytorch.serve.serve_model_wrapper") as mock_wrapper:
        # Test vLLM
        mock_wrapper.return_value = {"status": "running_pytorch_serve"}
        res = serve_module.serve_model("m")
        assert res["status"] == "running_vllm"

        app_factory = mock_wrapper.call_args[1]["app_factory"]
        with patch("gemma_4_sql.backends.pytorch.serve._create_vllm_app") as mock_create_vllm:
            app_factory()
            mock_create_vllm.assert_called_once()

        # Test native
        mock_wrapper.return_value = {"status": "running_pytorch_serve"}
        res = serve_module.serve_model("m", native_fallback=True)
        assert res["status"] == "running_pytorch_serve"

        app_factory_native = mock_wrapper.call_args[1]["app_factory"]
        with patch("gemma_4_sql.backends.pytorch.serve._create_native_app") as mock_create_native:
            app_factory_native()
            mock_create_native.assert_called_once()

        mock_wrapper.return_value = {"status": "failed"}
        res = serve_module.serve_model("m", native_fallback=True)
        assert res["status"] == "failed"

        # Test vllm failed
        mock_wrapper.return_value = {"status": "failed"}
        res = serve_module.serve_model("m")
        assert res["status"] == "failed"

        app_factory = mock_wrapper.call_args[1]["app_factory"]
        with patch("gemma_4_sql.backends.pytorch.serve._create_vllm_app") as mock_create_vllm2:
            app_factory()
            mock_create_vllm2.assert_called_once()

    # Test vLLM missing deps
    serve_module.AsyncEngineArgs = None
    with pytest.raises(DependencyMissingError):
        serve_module.serve_model("m")


@pytest.mark.asyncio
async def test_create_vllm_app_empty_generator(monkeypatch):
    """Docstring for test_create_vllm_app_empty_generator."""
    from unittest.mock import MagicMock

    import gemma_4_sql.backends.pytorch.serve as serve_module

    mock_fastapi = MagicMock()
    mock_app = MagicMock()
    mock_fastapi.__call__ = MagicMock(return_value=mock_app)

    monkeypatch.setattr(serve_module, "FastAPI", mock_fastapi)
    monkeypatch.setattr(serve_module, "AsyncEngineArgs", MagicMock())
    monkeypatch.setattr(serve_module, "AsyncLLMEngine", MagicMock())
    monkeypatch.setattr(serve_module, "JSONResponse", MagicMock())

    def mock_decorator(path):
        """Docstring for mock_decorator."""

        def wrapper(func):
            """Docstring for wrapper."""
            mock_app.generate_func = func
            return func

        return wrapper

    mock_app.post.side_effect = mock_decorator

    serve_module._create_vllm_app("m", 1)
    generate_fn = mock_app.generate_func

    mock_req = MagicMock()

    async def mock_json():
        """Docstring for mock_json."""
        return {"prompt": "p"}

    mock_req.json = mock_json

    async def mock_disconnected():
        """Docstring for mock_disconnected."""
        return False

    mock_req.is_disconnected = mock_disconnected

    mock_engine = MagicMock()
    serve_module.AsyncLLMEngine.from_engine_args.return_value = mock_engine

    async def mock_generate_stream(*args, **kwargs):
        # Empty generator
        """Docstring for mock_generate_stream."""
        if False:
            yield None

    mock_engine.generate.return_value = mock_generate_stream()

    def mock_json_response(content):
        """Docstring for mock_json_response."""
        return {"response": content}

    serve_module.JSONResponse.__call__ = mock_json_response

    res = await generate_fn(mock_req)
    assert res == {"response": {"sql": ""}}

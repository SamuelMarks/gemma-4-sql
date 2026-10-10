"""Extra PyTorch serving tests."""

from unittest.mock import MagicMock

import pytest

import gemma_4_sql.backends.pytorch.serve as serve_module


@pytest.mark.asyncio
async def test_vllm_app_truly_empty_generator(monkeypatch):
    """Test vLLM app generator disconnect handling."""
    mock_fastapi = MagicMock()
    mock_app = MagicMock()
    mock_fastapi.__call__ = MagicMock(return_value=mock_app)

    monkeypatch.setattr(serve_module, "FastAPI", mock_fastapi)
    monkeypatch.setattr(serve_module, "AsyncEngineArgs", MagicMock())
    monkeypatch.setattr(serve_module, "AsyncLLMEngine", MagicMock())
    monkeypatch.setattr(serve_module, "JSONResponse", MagicMock())

    def mock_decorator(path):
        """Mock decorator for route."""

        def wrapper(func):
            """Mock wrapper for route."""
            mock_app.generate_func = func
            return func

        return wrapper

    mock_app.post.side_effect = mock_decorator
    serve_module._create_vllm_app("m", 1)

    mock_req = MagicMock()

    async def mock_json():
        """Mock request json."""
        return {"prompt": "p"}

    mock_req.json = mock_json

    async def mock_disconnected():
        """Mock request disconnect."""
        return False

    mock_req.is_disconnected = mock_disconnected

    mock_engine = MagicMock()
    serve_module.AsyncLLMEngine.from_engine_args.return_value = mock_engine

    class EmptyAsyncGen:
        """Mock empty async generator."""

        def __aiter__(self):
            """Return self as iterator."""
            return self

        async def __anext__(self):
            """Raise StopAsyncIteration."""
            raise StopAsyncIteration

    mock_engine.generate.return_value = EmptyAsyncGen()

    def mock_json_response(content):
        """Mock JSON response."""
        return {"response": content}

    serve_module.JSONResponse.__call__ = mock_json_response

    res = await mock_app.generate_func(mock_req)
    assert res == {"response": {"sql": ""}}

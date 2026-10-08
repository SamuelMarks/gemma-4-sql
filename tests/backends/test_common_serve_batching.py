"""Module docstring."""

import asyncio
from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from gemma_4_sql.backends.common_serve import (
    GenerateRequest,
    create_common_app,
    serve_model_wrapper,
)


@pytest.fixture
def sample_dict():
    """Docstring for sample_dict."""
    return {"prompt": "SELECT *", "max_tokens": 50, "temperature": 0.5, "image_base64": "img", "audio_base64": "aud", "image_path": "path/img", "audio_path": "path/aud", "modality": "multimodal"}


def test_generate_request_from_dict_valid(sample_dict):
    """Docstring for test_generate_request_from_dict_valid."""
    req = GenerateRequest.from_dict(sample_dict)
    assert req.prompt == "SELECT *"
    assert req.max_tokens == 50


def test_generate_request_from_dict_defaults():
    """Docstring for test_generate_request_from_dict_defaults."""
    req = GenerateRequest.from_dict({"prompt": "test"})
    assert req.prompt == "test"
    assert req.modality == "text"


def test_generate_request_from_dict_legacy_keys():
    """Docstring for test_generate_request_from_dict_legacy_keys."""
    req = GenerateRequest.from_dict({"prompt": "test", "image": "img1", "audio": "aud1"})
    assert req.image_base64 == "img1"


def test_generate_request_from_dict_missing_prompt():
    """Docstring for test_generate_request_from_dict_missing_prompt."""
    with pytest.raises(ValueError):
        GenerateRequest.from_dict({"not_prompt": "test"})
    with pytest.raises(ValueError):
        GenerateRequest.from_dict({"prompt": 123})


def test_create_common_app_require_handlers():
    """Docstring for test_create_common_app_require_handlers."""
    with pytest.raises(ValueError):
        create_common_app("test_backend", "test_model", require_handlers=True)


def test_create_common_app_startup_callback():
    """Docstring for test_create_common_app_startup_callback."""
    startup_mock = MagicMock()
    create_common_app("test_backend", "test_model", startup_callback=startup_mock)
    startup_mock.assert_called_once()


def test_health_ready_list_models():
    """Docstring for test_health_ready_list_models."""
    app = create_common_app("test", "model")
    client = TestClient(app)

    res = client.get("/health")
    assert res.json()["status"] == "healthy"

    res = client.get("/ready")
    assert res.json()["status"] == "ready"

    res = client.get("/v1/models")
    assert res.json()["object"] == "list"


@pytest.mark.asyncio
async def test_generate_continuous_batching_generate_logic():
    """Docstring for test_generate_continuous_batching_generate_logic."""
    app = create_common_app("test", "model", generate_logic=lambda p: f"SYNC_{p}")
    client = TestClient(app)
    res = client.post("/generate", json={"prompt": "hello"})
    assert res.json()["sql"] == "SYNC_hello"


@pytest.mark.asyncio
async def test_generate_continuous_batching_batch_generate_logic():
    """Docstring for test_generate_continuous_batching_batch_generate_logic."""
    app = create_common_app("test", "model", batch_generate_logic=lambda p: [f"BATCH_{x}" for x in p], max_wait_ms=1.0)
    client = TestClient(app)
    res = client.post("/generate", json={"prompt": "world"})
    assert res.json()["sql"] == "BATCH_world"


def test_generate_multimodal_logic():
    """Docstring for test_generate_multimodal_logic."""
    app = create_common_app("test", "model", batch_generate_logic=lambda p: [f"B_{x}" for x in p], max_wait_ms=1.0)
    client = TestClient(app)
    with patch("gemma_4_sql.backends.common_multimodal.format_multimodal_prompt") as mock_fmt:
        mock_fmt.return_value = {"prompt": "MM_PROMPT"}
        res = client.post("/generate", json={"prompt": "hello", "image_base64": "XYZ"})
        assert res.json()["sql"] == "B_MM_PROMPT"


def test_generate_batching_error_propagation():
    """Docstring for test_generate_batching_error_propagation."""

    def err_logic(p):
        """Docstring for err_logic."""
        raise ValueError("err")

    app = create_common_app("test", "model", batch_generate_logic=err_logic, max_wait_ms=1.0)
    client = TestClient(app)
    with pytest.raises(ValueError):
        client.post("/generate", json={"prompt": "world"})


def test_generate_batching_mismatch_results():
    """Docstring for test_generate_batching_mismatch_results."""
    app = create_common_app("test", "model", batch_generate_logic=lambda p: ["EXTRA", "EXTRA"], max_wait_ms=1.0)
    client = TestClient(app)
    with pytest.raises(ValueError):
        client.post("/generate", json={"prompt": "world"})


def test_generate_no_logic():
    """Docstring for test_generate_no_logic."""
    app = create_common_app("test", "model")
    client = TestClient(app)
    with pytest.raises(NotImplementedError):
        client.post("/generate", json={"prompt": "world"})


def test_serve_model_wrapper_missing_deps():
    """Docstring for test_serve_model_wrapper_missing_deps."""
    res = serve_model_wrapper("test", "model", 8080, 32, True, "Deps missing", lambda: None)
    assert res["status"] == "Deps missing"


def test_serve_model_wrapper_success():
    """Docstring for test_serve_model_wrapper_success."""
    with patch("gemma_4_sql.backends.common_serve.uvicorn.run") as mock_run:
        res = serve_model_wrapper("test", "model", 8080, 32, False, "", lambda: MagicMock(), run_server=True)
        assert "running" in res["status"]
        mock_run.assert_called_once()

        # Test run_server = False
        res2 = serve_model_wrapper("test", "model", 8080, 32, False, "", lambda: MagicMock(), run_server=False)
        assert "running" in res2["status"]


def test_serve_model_wrapper_factory_error():
    """Docstring for test_serve_model_wrapper_factory_error."""

    def mock_factory():
        """Docstring for mock_factory."""
        raise RuntimeError("Factory failed")

    res = serve_model_wrapper("test", "model", 8080, 32, False, "", mock_factory)
    assert "failed" in res["status"]


@pytest.mark.asyncio
async def test_worker_cancel_and_fallback():
    """Docstring for test_worker_cancel_and_fallback."""
    with patch("gemma_4_sql.backends.common_serve.JSONResponse", None):
        app = create_common_app("test", "model", generate_logic=lambda p: f"SYNC_{p}")
        client = TestClient(app)
        res = client.post("/generate", json={"prompt": "fallback_test"})
        assert res.json()["sql"] == "SYNC_fallback_test"
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 200
        assert client.get("/v1/models").status_code == 200


# Testing Asyncio specifics (batching multiple, cancellation, RuntimeError fallback)
@pytest.mark.asyncio
async def test_generate_async_batching():
    """Docstring for test_generate_async_batching."""

    # Delay processing so multiple items get into queue
    async def slow_batch(prompts):
        """Docstring for slow_batch."""
        await asyncio.sleep(0.05)
        return [f"BATCH_{p}" for p in prompts]

    def sync_batch(prompts):
        """Docstring for sync_batch."""
        # We need this to block briefly or just return
        return [f"BATCH_{p}" for p in prompts]

    app = create_common_app("test", "model", batch_generate_logic=sync_batch, max_batch_size=2, max_wait_ms=50.0)

    # We use ASGITransport to hit the app concurrently
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # Fire two requests concurrently
        t1 = client.post("/generate", json={"prompt": "one"})
        t2 = client.post("/generate", json={"prompt": "two"})
        res1, res2 = await asyncio.gather(t1, t2)
        assert res1.json()["sql"] == "BATCH_one"
        assert res2.json()["sql"] == "BATCH_two"


@pytest.mark.asyncio
async def test_generate_cancellation():
    """Docstring for test_generate_cancellation."""

    async def cancel_later():
        """Docstring for cancel_later."""
        await asyncio.sleep(0.01)
        raise asyncio.CancelledError()

    app = create_common_app("test", "model", generate_logic=lambda p: "A")
    # To test cancel, we need to cancel the HTTP request
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # We can simulate cancel by adding a timeout using asyncio.wait_for
        import asyncio

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(client.post("/generate", json={"prompt": "cancel"}), timeout=0.001)

"""Module docstring."""

import asyncio

import pytest
from fastapi.testclient import TestClient

from gemma_4_sql.backends.common_serve import GenerateRequest, create_common_app, serve_model_wrapper


def test_generate_request_validation():
    """Docstring for test_generate_request_validation."""
    with pytest.raises(ValueError, match="Field 'prompt' must be a valid string"):
        GenerateRequest.from_dict({"not_prompt": "test"})
    with pytest.raises(ValueError, match="Field 'prompt' must be a valid string"):
        GenerateRequest.from_dict({"prompt": 123})


def test_require_handlers():
    """Docstring for test_require_handlers."""
    with pytest.raises(ValueError, match="At least one generation logic callback must be provided"):
        create_common_app("pytorch", "test", require_handlers=True)


def test_common_serve_endpoints():
    """Docstring for test_common_serve_endpoints."""
    app = create_common_app("pytorch", "test", generate_logic=lambda p: "SELECT 1;")
    client = TestClient(app)

    with pytest.MonkeyPatch.context() as mp:
        import asyncio

        original_get = asyncio.get_running_loop

        def fake_get():
            """Docstring for fake_get."""
            import inspect

            f = inspect.currentframe()
            if f and f.f_back and "common_serve.py" in f.f_back.f_code.co_filename:
                raise RuntimeError("no loop")
            return original_get()

        mp.setattr(asyncio, "get_running_loop", fake_get)

        resp = client.post("/generate", json={"prompt": "sync"})
        assert resp.json()["sql"] == "SELECT 1;"

    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 200
    assert client.get("/v1/models").status_code == 200


@pytest.mark.asyncio
async def test_worker_async():
    """Docstring for test_worker_async."""
    import httpx

    def b_gen(prompts):
        """Docstring for b_gen."""
        if prompts[0] == "mismatch":
            return []
        return ["R" * len(p) for p in prompts]

    app = create_common_app("pytorch", "test", batch_generate_logic=b_gen, max_batch_size=2)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        t1 = asyncio.create_task(client.post("/generate", json={"prompt": "a"}))
        t2 = asyncio.create_task(client.post("/generate", json={"prompt": "bb"}))
        r1, r2 = await asyncio.gather(t1, t2)
        assert r1.json()["sql"] == "R"
        assert r2.json()["sql"] == "RR"

        with pytest.raises(ValueError, match="Batch generation returned"):
            await client.post("/generate", json={"prompt": "mismatch"})


@pytest.mark.asyncio
async def test_worker_async_generate_fallback():
    """Docstring for test_worker_async_generate_fallback."""
    import httpx

    app = create_common_app("pytorch", "test", generate_logic=lambda p: "F", max_batch_size=2)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/generate", json={"prompt": "a"})
        assert r.json()["sql"] == "F"


@pytest.mark.asyncio
async def test_worker_async_both_none():
    """Docstring for test_worker_async_both_none."""
    import httpx

    app = create_common_app("pytorch", "test", max_batch_size=2)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        with pytest.raises(NotImplementedError):
            await client.post("/generate", json={"prompt": "a"})


@pytest.mark.asyncio
async def test_multimodal_request():
    """Docstring for test_multimodal_request."""
    import httpx

    app = create_common_app("pytorch", "test", generate_logic=lambda p: p)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/generate", json={"prompt": "p", "image_base64": "img"})
        assert "<image>" in r.json()["sql"]


def test_serve_model_wrapper():
    """Docstring for test_serve_model_wrapper."""
    res = serve_model_wrapper("pytorch", "test", max_batch_size=32, missing_deps=None, missing_status=None, app_factory=lambda: create_common_app("pytorch", "test"), run_server=False, port=8000)
    assert res["status"] == "running_pytorch_serve"

    def fail_factory():
        """Docstring for fail_factory."""
        raise RuntimeError("failed factory")

    res2 = serve_model_wrapper("pytorch", "test", max_batch_size=32, missing_deps=None, missing_status=None, app_factory=fail_factory, run_server=False, port=8000)
    assert "failed:" in res2["status"]

    res3 = serve_model_wrapper("pytorch", "test", max_batch_size=32, missing_deps=RuntimeError("deps"), missing_status="missing_pytorch", app_factory=fail_factory, run_server=False, port=8000)
    assert res3["status"] == "missing_pytorch"


def test_json_response_fallback():
    """Docstring for test_json_response_fallback."""
    import gemma_4_sql.backends.common_serve as mod

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(mod, "JSONResponse", None)
        app = create_common_app("pytorch", "test", generate_logic=lambda p: "SELECT 1;")

        import asyncio

        original_get = asyncio.get_running_loop

        def fake_get():
            """Docstring for fake_get."""
            import inspect

            f = inspect.currentframe()
            if f and f.f_back and "common_serve.py" in f.f_back.f_code.co_filename:
                raise RuntimeError("no loop")
            return original_get()

        mp.setattr(asyncio, "get_running_loop", fake_get)

        client = TestClient(app)

        r1 = client.post("/generate", json={"prompt": "sync"})
        assert r1.json()["sql"] == "SELECT 1;"
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 200
        assert client.get("/v1/models").status_code == 200


def test_sync_fallback_branches():
    """Docstring for test_sync_fallback_branches."""
    import asyncio

    with pytest.MonkeyPatch.context() as mp:
        original_get = asyncio.get_running_loop

        def fake_get():
            """Docstring for fake_get."""
            import inspect

            f = inspect.currentframe()
            if f and f.f_back and "common_serve.py" in f.f_back.f_code.co_filename:
                raise RuntimeError("no loop")
            return original_get()

        mp.setattr(asyncio, "get_running_loop", fake_get)

        # batch_generate_logic fallback
        app1 = create_common_app("pytorch", "test", batch_generate_logic=lambda p: ["B1"])
        c1 = TestClient(app1)
        assert c1.post("/generate", json={"prompt": "a"}).json()["sql"] == "B1"

        # No logic fallback
        app2 = create_common_app("pytorch", "test")
        c2 = TestClient(app2)
        with pytest.raises(NotImplementedError):
            c2.post("/generate", json={"prompt": "a"})


@pytest.mark.asyncio
async def test_common_serve_cancel_request():
    """Docstring for test_common_serve_cancel_request."""
    import asyncio

    import httpx

    def slow_batch(p):
        """Docstring for slow_batch."""
        import time

        time.sleep(0.1)
        return ["SUCCESS"]

    app = create_common_app("pytorch", "test", batch_generate_logic=slow_batch)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        task = asyncio.create_task(client.post("/generate", json={"prompt": "test"}))
        await asyncio.sleep(0.01)  # Give it time to hit the endpoint and queue
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(0.2)

    called = []
    create_common_app("pytorch", "test", startup_callback=lambda: called.append(1), generate_logic=lambda p: "")
    assert len(called) == 1


@pytest.mark.asyncio
async def test_common_serve_cancel_request_exception():
    """Docstring for test_common_serve_cancel_request_exception."""
    import asyncio

    import httpx

    from gemma_4_sql.backends.common_serve import create_common_app

    def error_batch(p):
        """Docstring for error_batch."""
        import time

        time.sleep(0.1)
        raise RuntimeError("Fail")

    app = create_common_app("pytorch", "test", batch_generate_logic=error_batch, max_batch_size=2)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        task = asyncio.create_task(client.post("/generate", json={"prompt": "test"}))
        await asyncio.sleep(0.01)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(0.2)

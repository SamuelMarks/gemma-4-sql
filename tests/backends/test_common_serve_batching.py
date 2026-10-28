"""Unit tests for FastAPI continuous batching server, request validation, and fallbacks."""

from __future__ import annotations

import asyncio
from typing import Never

import pytest
from fastapi.testclient import TestClient

from gemma_4_sql.backends.common_serve import (
    DirectRunnableAppProtocol,
    GenerateRequest,
    create_common_app,
    serve_model_wrapper,
)
from gemma_4_sql.type_hints import FastAPIAppProtocol, JSONDict, JSONValue


def test_generate_request_validation() -> None:
    """Verify generate request validation behavior."""
    with pytest.raises(ValueError, match="Field 'prompt' must be a valid string"):
        GenerateRequest.from_dict({"not_prompt": "test"})
    with pytest.raises(ValueError, match="Field 'prompt' must be a valid string"):
        GenerateRequest.from_dict({"prompt": 123})


def test_require_handlers() -> None:
    """Verify require handlers behavior."""
    with pytest.raises(ValueError, match="At least one generation logic callback must be provided"):
        create_common_app("pytorch", "test", require_handlers=True)


def test_common_serve_endpoints() -> None:
    """Verify common serve endpoints behavior and lifespan worker cleanup."""
    app = create_common_app("pytorch", "test", generate_logic=lambda _p: "SELECT 1;")  # pyright: ignore[reportUnknownLambdaType]
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 200
        assert client.get("/v1/models").status_code == 200

    with TestClient(app) as client:
        resp_async = client.post("/generate", json={"prompt": "async"})
        assert resp_async.json()["sql"] == "SELECT 1;"


@pytest.mark.asyncio
async def test_worker_async() -> None:
    """Verify worker async batches concurrent requests while running batch_generate_logic in worker thread."""
    import httpx

    observed_batch_sizes: list[int] = []

    def b_gen(prompts: list[str]) -> list[str]:
        """Provide b gen test helper recording batch size."""
        observed_batch_sizes.append(len(prompts))
        if prompts[0] == "mismatch":
            return []
        return ["R" * len(p) for p in prompts]

    app = create_common_app("pytorch", "test", batch_generate_logic=b_gen, max_batch_size=2, max_wait_ms=15.0)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        t1 = asyncio.create_task(client.post("/generate", json={"prompt": "a"}))
        t2 = asyncio.create_task(client.post("/generate", json={"prompt": "bb"}))
        r1, r2 = await asyncio.gather(t1, t2)
        assert r1.json()["sql"] == "R"
        assert r2.json()["sql"] == "RR"
        assert 2 in observed_batch_sizes

        with pytest.raises(ValueError, match="Batch generation returned"):
            await client.post("/generate", json={"prompt": "mismatch"})


@pytest.mark.asyncio
async def test_worker_async_generate_fallback() -> None:
    """Verify worker async generate fallback behavior."""
    import httpx

    app = create_common_app("pytorch", "test", generate_logic=lambda _p: "F", max_batch_size=2)  # pyright: ignore[reportUnknownLambdaType]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/generate", json={"prompt": "a"})
        assert r.json()["sql"] == "F"


@pytest.mark.asyncio
async def test_worker_async_both_none() -> None:
    """Verify worker async both none behavior."""
    import httpx

    app = create_common_app("pytorch", "test", max_batch_size=2)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        with pytest.raises(NotImplementedError):
            await client.post("/generate", json={"prompt": "a"})


@pytest.mark.asyncio
async def test_multimodal_request() -> None:
    """Verify multimodal request and sampling parameters are forwarded to single and backend handlers."""
    import httpx

    from gemma_4_sql.backends.common_serve import create_backend_serve_app

    app = create_common_app("pytorch", "test", generate_logic=lambda p: p)  # pyright: ignore[reportUnknownLambdaType]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/generate", json={"prompt": "p", "image_base64": "img"})
        assert "<image>" in r.json()["sql"]

    captured: JSONDict = {}

    def _kw_gen(model_name: str, prompt: str, **kwargs: JSONValue) -> dict[str, str]:
        """Record forwarded multimodal and sampling kwargs."""
        captured["model"] = model_name
        captured["prompt"] = prompt
        captured.update(kwargs)
        return {"sql": "SELECT 1"}

    app_kw = create_backend_serve_app("jax", "test-model", generate_sql_fn=_kw_gen)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_kw), base_url="http://test") as client:
        r_kw = await client.post(
            "/generate",
            json={
                "prompt": "query",
                "image_base64": "img_b64",
                "audio_base64": "aud_b64",
                "image_path": "schema.png",
                "audio_path": "voice.wav",
                "modality": "multimodal",
                "temperature": 0.4,
                "max_tokens": 64,
            },
        )
        assert r_kw.json()["sql"] == "SELECT 1"
        assert captured["image_bytes"] == "img_b64"
        assert captured["audio_clip"] == "aud_b64"
        assert captured["image_path"] == "schema.png"
        assert captured["audio_path"] == "voice.wav"
        assert captured["modality"] == "multimodal"
        assert captured["temperature"] == 0.4
        assert captured["max_length"] == 64

    app_strict = create_backend_serve_app(
        "jax",
        "test-model",
        generate_sql_fn=lambda model_name, prompt: {"sql": f"{model_name}:{prompt}"},  # pyright: ignore[reportUnknownLambdaType]
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_strict), base_url="http://test") as client:
        r_strict = await client.post("/generate", json={"prompt": "q", "temperature": 0.5})
        assert r_strict.json()["sql"] == "test-model:q"


def test_serve_model_wrapper() -> None:
    """Verify serve model wrapper behavior."""
    res = serve_model_wrapper(
        "pytorch",
        "test",
        max_batch_size=32,
        app_factory=lambda: create_common_app("pytorch", "test"),
        run_server=False,
        port=8000,
    )
    assert res["status"] == "running_pytorch_serve"

    def fail_factory() -> FastAPIAppProtocol:
        """Provide fail factory test helper."""
        raise RuntimeError("failed factory")

    res2 = serve_model_wrapper(
        "pytorch",
        "test",
        max_batch_size=32,
        app_factory=fail_factory,
        run_server=False,
        port=8000,
    )
    assert "failed:" in res2["status"]

    from gemma_4_sql.exceptions import ServingError

    with pytest.raises(ServingError, match="failed factory"):
        serve_model_wrapper(
            "pytorch",
            "test",
            max_batch_size=32,
            app_factory=fail_factory,
            run_server=False,
            port=8000,
            raise_on_error=True,
        )


def test_json_response_fallback() -> None:
    """Verify json response fallback behavior when JSONResponse is None."""
    import gemma_4_sql.backends.common_serve as mod

    with pytest.MonkeyPatch.context() as monkeypatch_ctx:
        monkeypatch_ctx.setattr(mod, "JSONResponse", None)
        app = create_common_app("pytorch", "test", generate_logic=lambda _p: "SELECT 1;")  # pyright: ignore[reportUnknownLambdaType]
        client = TestClient(app)

        r1 = client.post("/generate", json={"prompt": "sync"})
        assert r1.json()["sql"] == "SELECT 1;"
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 200
        assert client.get("/v1/models").status_code == 200


def test_sync_fallback_branches() -> None:
    """Verify `app.run_direct` synchronous helper for single, batched, and missing handlers."""
    app0: DirectRunnableAppProtocol = create_common_app("pytorch", "test", generate_logic=lambda p: f"G:{p}")  # pyright: ignore[reportUnknownLambdaType]
    assert app0.run_direct("q") == "G:q"

    app1: DirectRunnableAppProtocol = create_common_app("pytorch", "test", batch_generate_logic=lambda p: [f"B:{p[0]}"])
    assert app1.run_direct("q") == "B:q"

    app2: DirectRunnableAppProtocol = create_common_app("pytorch", "test")
    with pytest.raises(NotImplementedError):
        app2.run_direct("q")


@pytest.mark.asyncio
async def test_common_serve_cancel_request() -> None:
    """Verify common serve cancel request behavior using deterministic event synchronization."""
    import threading

    import httpx

    release_worker = threading.Event()
    worker_started = threading.Event()

    def slow_batch(_p: list[str]) -> list[str]:
        """Signal worker start and wait for release_worker event."""
        worker_started.set()
        release_worker.wait(timeout=1.0)
        return ["SUCCESS"]

    app = create_common_app("pytorch", "test", batch_generate_logic=slow_batch, max_wait_ms=1.0)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        task = asyncio.create_task(client.post("/generate", json={"prompt": "test"}))
        while not worker_started.is_set():
            await asyncio.sleep(0.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release_worker.set()
        await asyncio.sleep(0.01)

    called: list[int] = []
    create_common_app("pytorch", "test", startup_callback=lambda: called.append(1), generate_logic=lambda _p: "")  # pyright: ignore[reportUnknownLambdaType]
    assert len(called) == 1


@pytest.mark.asyncio
async def test_common_serve_cancel_request_exception() -> None:
    """Verify common serve cancel request exception behavior using deterministic event synchronization."""
    import threading

    import httpx

    release_worker = threading.Event()
    worker_started = threading.Event()

    def error_batch(_p: list[str]) -> Never:
        """Signal worker start, wait for release_worker, then raise RuntimeError."""
        worker_started.set()
        release_worker.wait(timeout=1.0)
        raise RuntimeError("Fail")

    app = create_common_app("pytorch", "test", batch_generate_logic=error_batch, max_batch_size=2, max_wait_ms=1.0)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        task = asyncio.create_task(client.post("/generate", json={"prompt": "test"}))
        while not worker_started.is_set():
            await asyncio.sleep(0.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release_worker.set()
        await asyncio.sleep(0.01)


def test_fastapi_missing_and_run_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify serve_model_wrapper raises when FastAPI/uvicorn is None and calls uvicorn.run when run_server=True."""
    import gemma_4_sql.backends.common_serve as mod
    from gemma_4_sql.exceptions import DependencyMissingError

    real_app = create_common_app("pytorch", "test", generate_logic=lambda _p: "SELECT 1")  # pyright: ignore[reportUnknownLambdaType]
    monkeypatch.setattr(mod, "FastAPI", None)
    with pytest.raises(DependencyMissingError):
        mod.serve_model_wrapper(
            backend_name="pytorch",
            model_name="test",
            max_batch_size=32,
            app_factory=lambda: real_app,
            port=8000,
        )

    monkeypatch.undo()
    called: list[bool] = []

    class MockUvicorn:
        """Stub uvicorn module."""

        def run(self, _app: FastAPIAppProtocol, *, host: str = "0.0.0.0", port: int = 8000) -> None:
            """Record server execution."""
            _ = (host, port)
            called.append(True)

    monkeypatch.setattr(mod, "uvicorn", MockUvicorn())
    mod.serve_model_wrapper(
        backend_name="pytorch",
        model_name="test",
        max_batch_size=32,
        app_factory=lambda: real_app,
        run_server=True,
        port=8000,
    )
    assert len(called) == 1

    from gemma_4_sql.exceptions import InferenceError

    app_ok = mod.create_backend_serve_app("jax", "test", generate_sql_fn=lambda **_k: {"sql": "SELECT 1"})  # pyright: ignore[reportUnknownLambdaType]
    client_ok = TestClient(app_ok)
    assert client_ok.post("/generate", json={"prompt": "hi"}).json()["sql"] == "SELECT 1"

    app_empty = mod.create_backend_serve_app("jax", "test", generate_sql_fn=lambda **_k: {"sql": ""})  # pyright: ignore[reportUnknownLambdaType]
    client_empty = TestClient(app_empty, raise_server_exceptions=False)
    assert client_empty.post("/generate", json={"prompt": "hi"}).status_code == 500

    def _raise_val(**_k: JSONValue) -> dict[str, str]:
        """Raise ValueError to simulate backend failure."""
        raise ValueError("boom")

    app_err = mod.create_backend_serve_app("jax", "test", generate_sql_fn=_raise_val)
    client_err = TestClient(app_err, raise_server_exceptions=False)
    assert client_err.post("/generate", json={"prompt": "hi"}).status_code == 500
    assert InferenceError("test").backend is None

    with pytest.raises(DependencyMissingError, match="missing serve dep"):
        mod.serve_model_wrapper(
            backend_name="pytorch",
            model_name="test",
            max_batch_size=32,
            app_factory=lambda: (_ for _ in ()).throw(DependencyMissingError("missing serve dep")),
            port=8000,
        )

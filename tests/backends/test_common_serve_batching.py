"""Tests for common FastAPI serving continuous batching engine and endpoints."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from gemma_4_sql.backends.common_serve import create_common_app


class MockRequest:
    """Mock HTTP request object."""

    def __init__(self, data: dict[str, Any]) -> None:
        """Initialize MockRequest.

        Args:
            data: Payload dictionary returned by json().
        """
        self._data = data

    async def json(self) -> dict[str, Any]:
        """Return payload json dictionary.

        Returns:
            The payload dictionary.
        """
        return self._data


@pytest.mark.asyncio
async def test_continuous_batching_bundle_and_resolve(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that concurrent requests are bundled and resolved by batching worker."""
    batches_received: list[list[str]] = []

    def mock_batch_gen(prompts: list[str]) -> list[str]:
        """Mock batch generation function.

        Args:
            prompts: List of prompt strings.

        Returns:
            List of generated SQL queries.
        """
        batches_received.append(prompts)
        return [f"SELECT count(*) FROM {p}" for p in prompts]

    app: Any = create_common_app(
        backend_name="pytorch",
        model_name="test-model",
        test_mode=False,
        batch_generate_logic=mock_batch_gen,
        max_batch_size=4,
        max_wait_ms=50.0,
    )

    generate_route = None
    health_route = None
    ready_route = None
    models_route = None
    for route in app.routes:
        if getattr(route, "path", None) == "/generate":
            generate_route = route.endpoint
        elif getattr(route, "path", None) == "/health":
            health_route = route.endpoint
        elif getattr(route, "path", None) == "/ready":
            ready_route = route.endpoint
        elif getattr(route, "path", None) == "/v1/models":
            models_route = route.endpoint

    assert generate_route is not None
    assert health_route is not None
    assert ready_route is not None
    assert models_route is not None

    # Test health, ready, and models endpoints
    health_res = await health_route()
    assert health_res.status_code == 200
    assert "healthy" in health_res.body.decode()

    ready_res = await ready_route()
    assert ready_res.status_code == 200
    assert "ready" in ready_res.body.decode()

    models_res = await models_route()
    assert models_res.status_code == 200
    assert "test-model" in models_res.body.decode()

    monkeypatch.setattr("gemma_4_sql.backends.common_serve.JSONResponse", None)
    ready_res_raw = await ready_route()
    assert ready_res_raw["status"] == "ready"
    monkeypatch.undo()

    # Dispatch concurrent generation requests
    req1 = MockRequest({"prompt": "users"})
    req2 = MockRequest({"prompt": "orders"})
    req3 = MockRequest({"prompt": "items"})
    req4 = MockRequest({"prompt": "payments"})

    res1, res2, res3, res4 = await asyncio.gather(
        generate_route(req1),
        generate_route(req2),
        generate_route(req3),
        generate_route(req4),
    )

    assert "SELECT count(*) FROM users" in res1.body.decode()
    assert "SELECT count(*) FROM orders" in res2.body.decode()
    assert "SELECT count(*) FROM items" in res3.body.decode()
    assert "SELECT count(*) FROM payments" in res4.body.decode()


@pytest.mark.asyncio
async def test_continuous_batching_worker_error_propagation() -> None:
    """Test that batch worker exceptions are propagated to waiting request futures."""

    def failing_batch_gen(_prompts: list[str]) -> list[str]:
        """Raise RuntimeError to simulate generation failure.

        Args:
            _prompts: Prompt list.

        Raises:
            RuntimeError: Simulated engine error.
        """
        msg = "Inference engine failed"
        raise RuntimeError(msg)

    app: Any = create_common_app(
        backend_name="jax",
        model_name="test-jax-model",
        test_mode=False,
        batch_generate_logic=failing_batch_gen,
        max_batch_size=2,
        max_wait_ms=10.0,
    )

    generate_route = None
    for route in app.routes:
        if getattr(route, "path", None) == "/generate":
            generate_route = route.endpoint
            break

    assert generate_route is not None

    req = MockRequest({"prompt": "fail_test"})
    with pytest.raises(RuntimeError, match="Inference engine failed"):
        await generate_route(req)


@pytest.mark.asyncio
async def test_common_serve_health_and_models_endpoints() -> None:
    """Test /health and /v1/models endpoints on the FastAPI server."""
    app: Any = create_common_app(
        backend_name="pytorch",
        model_name="test-pt-model",
        test_mode=True,
    )
    health_route = None
    models_route = None
    for route in app.routes:
        if getattr(route, "path", None) == "/health":
            health_route = route.endpoint
        elif getattr(route, "path", None) == "/v1/models":
            models_route = route.endpoint

    assert health_route is not None
    assert models_route is not None

    h_res = await health_route()
    assert h_res is not None

    m_res = await models_route()
    assert m_res is not None


@pytest.mark.asyncio
async def test_common_serve_edge_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test common_serve fallback and error execution branches."""
    from gemma_4_sql.backends.common_serve import serve_model_wrapper

    # 1. startup_callback invocation
    called = []
    create_common_app(
        backend_name="test",
        model_name="m1",
        startup_callback=lambda: called.append(True),
        test_mode=False,
    )
    assert len(called) == 1

    # 2. generate_logic fallback (without batch_generate_logic)
    app2: Any = create_common_app(
        backend_name="test",
        model_name="m2",
        generate_logic=lambda p: f"SINGLE: {p}",
        test_mode=False,
    )
    gen2 = next(r.endpoint for r in app2.routes if getattr(r, "path", None) == "/generate")
    res2 = await gen2(MockRequest({"prompt": "p2"}))
    assert "SINGLE: p2" in res2.body.decode()

    # 3. NotImplementedError when no generation logic is provided
    app3: Any = create_common_app(
        backend_name="custom_backend",
        model_name="m3",
        test_mode=False,
    )
    gen3 = next(r.endpoint for r in app3.routes if getattr(r, "path", None) == "/generate")
    with pytest.raises(NotImplementedError, match="No generation logic registered"):
        await gen3(MockRequest({"prompt": "p3"}))

    # 3b. Batch output length mismatch validation
    app3b: Any = create_common_app(
        backend_name="mismatch_backend",
        model_name="m3b",
        batch_generate_logic=lambda prompts: ["only_one_result"],
        test_mode=False,
    )
    gen3b = next(r.endpoint for r in app3b.routes if getattr(r, "path", None) == "/generate")
    task1 = asyncio.create_task(gen3b(MockRequest({"prompt": "req1"})))
    task2 = asyncio.create_task(gen3b(MockRequest({"prompt": "req2"})))
    with pytest.raises(ValueError, match="Batch generation returned"):
        await asyncio.gather(task1, task2)

    # 4. JSONResponse is None
    import gemma_4_sql.backends.common_serve as cs

    monkeypatch.setattr(cs, "JSONResponse", None)
    app4: Any = create_common_app(
        backend_name="test",
        model_name="m4",
        generate_logic=lambda p: f"FALLBACK {p}",
        test_mode=True,
    )
    gen4 = next(r.endpoint for r in app4.routes if getattr(r, "path", None) == "/generate")
    res4 = await gen4(MockRequest({"prompt": "p4"}))
    assert res4 == {"sql": "FALLBACK p4", "modality": "text"}

    h4 = next(r.endpoint for r in app4.routes if getattr(r, "path", None) == "/health")
    assert (await h4())["status"] == "healthy"

    m4 = next(r.endpoint for r in app4.routes if getattr(r, "path", None) == "/v1/models")
    assert (await m4())["object"] == "list"

    # 5. serve_model_wrapper exception
    def bad_factory() -> Any:
        """Execute bad factory helper."""
        msg = "failed to build app"
        raise RuntimeError(msg)

    res5 = serve_model_wrapper(
        backend_name="err",
        model_name="m",
        port=8000,
        max_batch_size=1,
        missing_deps=False,
        missing_status="",
        app_factory=bad_factory,
    )
    assert "failed" in res5["status"]
    assert res5["app"] is None

    # 6. Cancellation handling during generate
    app6: Any = create_common_app(backend_name="test", model_name="m6", test_mode=False)
    gen6 = next(r.endpoint for r in app6.routes if getattr(r, "path", None) == "/generate")
    task = asyncio.create_task(gen6(MockRequest({"prompt": "cancel_me"})))
    await asyncio.sleep(0.001)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # 7. worker_task failed/None fallback branch
    app7: Any = create_common_app(backend_name="test", model_name="m7", generate_logic=lambda p: f"SYNC: {p}", test_mode=True)
    monkeypatch.setattr(asyncio, "get_running_loop", lambda: (_ for _ in ()).throw(RuntimeError("no loop")))
    gen7 = next(r.endpoint for r in app7.routes if getattr(r, "path", None) == "/generate")
    res7 = await gen7(MockRequest({"prompt": "p7"}))
    sql_text7 = res7.body.decode() if hasattr(res7, "body") else str(res7)
    assert "SYNC: p7" in sql_text7

    # 8. worker_task None and generate_logic None raises NotImplementedError
    app8: Any = create_common_app(backend_name="nosync", model_name="m8", test_mode=True)
    gen8 = next(r.endpoint for r in app8.routes if getattr(r, "path", None) == "/generate")
    with pytest.raises(NotImplementedError, match="No generation logic registered"):
        await gen8(MockRequest({"prompt": "p8"}))

    # 9. worker_task None and batch_generate_logic provided fallback branch
    app9: Any = create_common_app(
        backend_name="test_batch_fallback",
        model_name="m9",
        batch_generate_logic=lambda ps: [f"BATCH_FALLBACK: {p}" for p in ps],
        test_mode=True,
    )
    gen9 = next(r.endpoint for r in app9.routes if getattr(r, "path", None) == "/generate")
    res9 = await gen9(MockRequest({"prompt": "p9"}))
    sql_text9 = res9.body.decode() if hasattr(res9, "body") else str(res9)
    assert "BATCH_FALLBACK: p9" in sql_text9

    # 10. item future already done when exception occurs in batch
    app10: Any = create_common_app(
        backend_name="test_predone_err",
        model_name="m10",
        max_wait_ms=50.0,
        batch_generate_logic=lambda ps: (_ for _ in ()).throw(ValueError("forced error")),
        test_mode=False,
    )
    gen10 = next(r.endpoint for r in app10.routes if getattr(r, "path", None) == "/generate")
    task_a = asyncio.create_task(gen10(MockRequest({"prompt": "p10_a"})))
    await asyncio.sleep(0.001)
    task_a.cancel()
    await asyncio.sleep(0.07)


@pytest.mark.asyncio
async def test_common_serve_ready_and_validation() -> None:
    """Test /ready route, require_handlers check, and GenerateRequest schema validation."""
    from gemma_4_sql.backends.common_serve import GenerateRequest

    # 1. GenerateRequest validation
    req = GenerateRequest.from_dict({"prompt": "SELECT 1", "max_tokens": 64, "temperature": 0.5})
    assert req.prompt == "SELECT 1"
    assert req.max_tokens == 64
    assert req.temperature == 0.5

    with pytest.raises(ValueError, match="Field 'prompt' must be a valid string."):
        GenerateRequest.from_dict({"prompt": None})

    # 2. require_handlers validation during app construction
    with pytest.raises(ValueError, match="At least one generation logic callback must be provided"):
        create_common_app(backend_name="strict_backend", model_name="m_strict", require_handlers=True)

    # 3. /ready endpoint
    app: Any = create_common_app(
        backend_name="ready_backend",
        model_name="m_ready",
        generate_logic=lambda p: f"SQL: {p}",
        test_mode=True,
    )
    ready_route = next(r.endpoint for r in app.routes if getattr(r, "path", None) == "/ready")
    res = await ready_route()
    body = res.body.decode() if hasattr(res, "body") else str(res)
    assert "ready" in body
    assert "ready_backend" in body
    assert "queue_depth" in body

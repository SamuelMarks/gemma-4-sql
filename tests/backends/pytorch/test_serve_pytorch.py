"""Tests for PyTorch continuous batching and vLLM serving without untyped mocks."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from unittest.mock import patch

from tests.conftest import requires_torch

pytestmark = requires_torch

import pytest

from gemma_4_sql.backends.common_serve import DirectRunnableAppProtocol
from gemma_4_sql.exceptions import DependencyMissingError, InferenceError
from gemma_4_sql.type_hints import FastAPIAppProtocol, JSONDict


class _FakeRequest:
    """Typed ASGI/FastAPI request double for vLLM route handler tests."""

    def __init__(self, payload: JSONDict, *, disconnected: bool = False) -> None:
        """Initialize payload and disconnect flag."""
        self._payload = payload
        self.disconnected = disconnected

    async def json(self) -> JSONDict:
        """Return JSON request payload."""
        return self._payload

    async def is_disconnected(self) -> bool:
        """Return `self.disconnected`."""
        return self.disconnected


type RouteHandler = Callable[[_FakeRequest], Awaitable[dict[str, JSONDict]]]


class _FakeFastAPIApp:
    """Typed FastAPI app double capturing `@app.post(...)` route handlers."""

    generate_func: RouteHandler | None = None

    def post(self, _path: str) -> Callable[[RouteHandler], RouteHandler]:
        """Register and return route handler.

        Returns:
            Route registration decorator.
        """

        def _decorator(func: RouteHandler) -> RouteHandler:
            """Store `func` on `self.generate_func`.

            Returns:
                Registered `func` handler unchanged.
            """
            self.generate_func = func
            return func

        return _decorator


class _MockCompletionOut:
    """Simulated vLLM completion output."""

    text: str = "sql"


class _MockRequestOutput:
    """Simulated vLLM request output."""

    outputs: tuple[_MockCompletionOut, ...] = (_MockCompletionOut(),)


class _EmptyAsyncGen:
    """Async iterator yielding zero items."""

    def __aiter__(self) -> _EmptyAsyncGen:
        """Return self."""
        return self

    async def __anext__(self) -> _MockRequestOutput:
        """Stop immediately.

        Raises:
            StopAsyncIteration: Always raised to terminate empty stream.
        """
        raise StopAsyncIteration


class _FakeVLLMEngine:
    """Typed `AsyncLLMEngine` double."""

    def __init__(self, *, empty: bool = False) -> None:
        """Initialize stream mode and `aborted_ids` list."""
        self.empty = empty
        self.aborted_ids: list[str] = []

    def generate(
        self, *_args: str | dict[str, int | float] | None, **_kwargs: str
    ) -> AsyncIterator[_MockRequestOutput]:
        """Return an async stream yielding one `_MockRequestOutput` or zero items."""
        if self.empty:
            return _EmptyAsyncGen()

        async def _stream() -> AsyncIterator[_MockRequestOutput]:
            """Yield single `_MockRequestOutput`."""
            yield _MockRequestOutput()

        return _stream()

    async def abort(self, request_id: str) -> None:
        """Record aborted `request_id`."""
        self.aborted_ids.append(request_id)


class _FakeAsyncArgs:
    """Typed `AsyncEngineArgs` constructor double."""

    def __init__(
        self,
        *,
        model: str,
        max_num_seqs: int = 256,
        max_num_batched_tokens: int = 4096,
        disable_log_requests: bool = True,
    ) -> None:
        """Store `model`, `max_num_seqs`, `max_num_batched_tokens`, and `disable_log_requests`."""
        self.model = model
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.disable_log_requests = disable_log_requests


@pytest.fixture(autouse=True)
def _restore_serve_mod(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore `serve_module` globals after each test."""
    import gemma_4_sql.backends.pytorch.serve as serve_module

    monkeypatch.setattr(serve_module, "torch", serve_module.torch)
    monkeypatch.setattr(serve_module, "FastAPI", serve_module.FastAPI)
    monkeypatch.setattr(serve_module, "JSONResponse", serve_module.JSONResponse)
    monkeypatch.setattr(serve_module, "AsyncEngineArgs", serve_module.AsyncEngineArgs)
    monkeypatch.setattr(serve_module, "AsyncLLMEngine", serve_module.AsyncLLMEngine)


def _wrap_json_response(*, content: JSONDict) -> dict[str, JSONDict]:
    """Wrap `content` dict in `{'response': content}`.

    Returns:
        Wrapped response dictionary.
    """
    return {"response": content}


@pytest.mark.asyncio
async def test_create_vllm_app(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `_create_vllm_app` registers `/generate`, streams vLLM outputs, and handles disconnects."""
    import gemma_4_sql.backends.pytorch.serve as serve_module

    fake_app = _FakeFastAPIApp()
    fake_engine = _FakeVLLMEngine(empty=False)

    class _FakeAsyncLLMCls:
        """Typed `AsyncLLMEngine` class double."""

        @classmethod
        def from_engine_args(cls, _args: _FakeAsyncArgs) -> _FakeVLLMEngine:
            """Return `fake_engine`."""
            return fake_engine

    def _make_fake_app(**_kw: str | float) -> _FakeFastAPIApp:
        """Return `fake_app`."""
        return fake_app

    monkeypatch.setattr(serve_module, "FastAPI", _make_fake_app)
    monkeypatch.setattr(serve_module, "AsyncEngineArgs", _FakeAsyncArgs)
    monkeypatch.setattr(serve_module, "AsyncLLMEngine", _FakeAsyncLLMCls)
    monkeypatch.setattr(serve_module, "JSONResponse", _wrap_json_response)

    app = serve_module._create_vllm_app("m", 1)
    assert app is fake_app
    generate_fn = fake_app.generate_func
    assert generate_fn is not None

    res = await generate_fn(_FakeRequest({"prompt": "p"}, disconnected=False))
    assert res == {"response": {"sql": "sql"}}

    # Client disconnect aborts vLLM request
    res2 = await generate_fn(_FakeRequest({"prompt": "p"}, disconnected=True))
    assert res2 == {"response": {"error": "Client disconnected"}}
    assert len(fake_engine.aborted_ids) == 1

    # Missing JSONResponse returns None
    monkeypatch.setattr(serve_module, "JSONResponse", None)
    assert serve_module._create_vllm_app("m", 1) is None


def test_create_native_app() -> None:
    """Verify `_create_native_app` builds a real FastAPI app and serves `/health` and `/generate` via `TestClient`."""
    import gemma_4_sql.backends.pytorch.serve as serve_module
    from tests.utils.test_utils import make_test_client

    app = serve_module._create_native_app("m", 16)
    assert isinstance(app, DirectRunnableAppProtocol)
    gen_ret: dict[str, str] = {"sql": "SELECT 1;"}
    raise_err = False

    def _fake_generate_sql(**_kwargs: str | int) -> dict[str, str]:
        """Return `gen_ret` or raise `RuntimeError`.

        Raises:
            RuntimeError: If `raise_err` is True.
        """
        if raise_err:
            raise RuntimeError("error")
        return gen_ret

    with patch("gemma_4_sql.backends.pytorch.inference.generate_sql", side_effect=_fake_generate_sql):
        assert app.run_direct("p") == "SELECT 1;"

        with make_test_client(app) as client:
            health_res = client.get("/health")
            assert health_res.status_code == 200
            assert health_res.json()["backend"] == "pytorch"

            gen_res = client.post("/generate", json={"prompt": "SELECT 1"})
            assert gen_res.status_code == 200
            assert gen_res.json()["sql"] == "SELECT 1;"

        gen_ret = {}
        with pytest.raises(InferenceError, match="PyTorch inference returned empty SQL"):
            app.run_direct("p")

        raise_err = True
        with pytest.raises(InferenceError, match="PyTorch generation failed"):
            app.run_direct("p")


def test_serve_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `serve_model` dispatches to vLLM or native PyTorch continuous batching through `serve_model_wrapper`."""
    import gemma_4_sql.backends.pytorch.serve as serve_module
    from gemma_4_sql.backends.common_serve import get_serve_handle
    from tests.utils.test_utils import make_test_client

    fake_engine = _FakeVLLMEngine(empty=False)

    class _FakeAsyncLLMCls:
        """Typed `AsyncLLMEngine` class double."""

        @classmethod
        def from_engine_args(cls, _args: _FakeAsyncArgs) -> _FakeVLLMEngine:
            """Return `fake_engine`."""
            return fake_engine

    monkeypatch.setattr(serve_module, "AsyncEngineArgs", _FakeAsyncArgs)
    monkeypatch.setattr(serve_module, "AsyncLLMEngine", _FakeAsyncLLMCls)

    res_vllm = serve_module.serve_model("m")
    assert res_vllm["status"] == "running_vllm"
    assert get_serve_handle(res_vllm).app is not None

    res_native = serve_module.serve_model("m", native_fallback=True)
    assert res_native["status"] == "running_pytorch_serve"
    native_app = get_serve_handle(res_native).app
    assert isinstance(native_app, FastAPIAppProtocol)
    with patch("gemma_4_sql.backends.pytorch.inference.generate_sql", return_value={"sql": "SELECT 42;"}):
        with make_test_client(native_app) as client:
            resp = client.post("/generate", json={"prompt": "test"})
            assert resp.status_code == 200
            assert resp.json()["sql"] == "SELECT 42;"

    with patch("gemma_4_sql.backends.pytorch.serve._create_native_app", side_effect=RuntimeError("native boom")):
        res_fail_native = serve_module.serve_model("m", native_fallback=True)
        assert res_fail_native["status"] == "failed: native boom"

    with patch("gemma_4_sql.backends.pytorch.serve._create_vllm_app", side_effect=RuntimeError("vllm boom")):
        res_fail_vllm = serve_module.serve_model("m")
        assert res_fail_vllm["status"] == "failed: vllm boom"

    monkeypatch.setattr(serve_module, "AsyncEngineArgs", None)
    with pytest.raises(DependencyMissingError):
        serve_module.serve_model("m")


@pytest.mark.asyncio
async def test_create_vllm_app_empty_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `_create_vllm_app` returns empty SQL when the vLLM stream yields zero items."""
    import gemma_4_sql.backends.pytorch.serve as serve_module

    fake_app = _FakeFastAPIApp()
    empty_engine = _FakeVLLMEngine(empty=True)

    class _FakeEmptyLLMCls:
        """Typed `AsyncLLMEngine` class double returning `empty_engine`."""

        @classmethod
        def from_engine_args(cls, _args: _FakeAsyncArgs) -> _FakeVLLMEngine:
            """Return `empty_engine`."""
            return empty_engine

    def _make_fake_app(**_kw: str | float) -> _FakeFastAPIApp:
        """Return `fake_app`."""
        return fake_app

    monkeypatch.setattr(serve_module, "FastAPI", _make_fake_app)
    monkeypatch.setattr(serve_module, "AsyncEngineArgs", _FakeAsyncArgs)
    monkeypatch.setattr(serve_module, "AsyncLLMEngine", _FakeEmptyLLMCls)
    monkeypatch.setattr(serve_module, "JSONResponse", _wrap_json_response)

    serve_module._create_vllm_app("m", 1)
    generate_fn = fake_app.generate_func
    assert generate_fn is not None

    res = await generate_fn(_FakeRequest({"prompt": "p"}, disconnected=False))
    assert res == {"response": {"sql": ""}}

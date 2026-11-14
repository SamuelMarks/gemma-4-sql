"""Behavioral and unit tests for the SDK Agent module using typed protocol test doubles."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest

from gemma_4_sql.sdk.agent import (
    AgentContext,
    _process_single_prompt,
    classify_error_hint,
    extract_sql_from_response,
    run_agentic_loop,
    run_agentic_loop_async,
)
from gemma_4_sql.sdk.db_engine import LiveDatabaseEngine
from gemma_4_sql.sdk.protocols import BackendProtocol
from gemma_4_sql.type_hints import AgentResultDict, InferenceResultDict, JSONPrimitive, JSONValue
from tests.utils.test_utils import RecordingBackendDouble


class _ConfigurableAgentBackend(RecordingBackendDouble):
    """Typed `BackendProtocol` test double supporting custom `generate_sql` callbacks and kwargs capture."""

    def __init__(
        self,
        *,
        sql_fn: Callable[[str, str], InferenceResultDict] | None = None,
        static_res: InferenceResultDict | None = None,
    ) -> None:
        """Initialize configurable backend double.

        Args:
            sql_fn: Optional callback `(model_name, prompt) -> InferenceResultDict`.
            static_res: Optional constant `InferenceResultDict` returned on every call.

        """
        super().__init__()
        self._sql_fn = sql_fn
        self._static_res = static_res
        self.last_kwargs: dict[str, JSONValue] = {}

    def generate_sql(
        self,
        model_name: str,
        prompt: str,
        beam_width: int = 3,
        max_length: int = 50,
        **kwargs: JSONValue,
    ) -> InferenceResultDict:
        """Record `kwargs` and return configured `InferenceResultDict`.

        Returns:
            Configured or default `InferenceResultDict`.

        """
        self.last_kwargs = {"beam_width": beam_width, "max_length": max_length, **kwargs}
        if self._sql_fn is not None:
            return self._sql_fn(model_name, prompt)
        if self._static_res is not None:
            return self._static_res
        return super().generate_sql(model_name, prompt, beam_width, max_length, **kwargs)


def _make_backend_getter(backend_double: BackendProtocol) -> Callable[[str], BackendProtocol]:
    """Create a typed `get_backend` replacement returning `backend_double`.

    Returns:
        Backend resolver callable returning `backend_double`.

    """

    def _get(_name: str) -> BackendProtocol:
        """Return `backend_double`."""
        return backend_double

    return _get


def test_extract_sql_from_response() -> None:
    """Verify `extract_sql_from_response` strips Markdown SQL fences and preserves raw SQL."""
    assert (
        extract_sql_from_response("""```sql
SELECT 1
```""")
        == "SELECT 1"
    )
    assert (
        extract_sql_from_response("""```
SELECT 1
```""")
        == "SELECT 1"
    )
    assert extract_sql_from_response("SELECT 1") == "SELECT 1"
    assert classify_error_hint(None) == ""
    assert classify_error_hint("") == ""


@pytest.mark.parametrize("backend", ["jax", "keras", "maxtext", "pytorch", "mlx"])
def test_agentic_loop_self_correction_against_real_sqlite(backend: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify multi-turn self-correction against a real seeded in-memory SQLite table across backends."""
    ddl = """
    CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER);
    INSERT INTO users (id, name, age) VALUES (1, 'Alice', 35), (2, 'Bob', 25);
    """

    call_count = 0

    def mock_generate(model_name: str, current_prompt: str) -> InferenceResultDict:
        """Simulate an initial invalid column query followed by a corrected valid SQL query.

        Returns:
            Simulated `InferenceResultDict` for the current turn.

        """
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {
                "backend": backend,
                "model": model_name,
                "prompt": current_prompt,
                "sql": "SELECT nonexistent_col FROM users WHERE age > 30;",
                "beam_width": 3,
                "status": "success",
                "confidence_score": 0.9,
            }
        assert "no such column: nonexistent_col" in current_prompt
        assert "[Hint: check valid column names]" in current_prompt
        return {
            "backend": backend,
            "model": model_name,
            "prompt": current_prompt,
            "sql": """```sql
SELECT name FROM users WHERE age > 30;
```""",
            "beam_width": 3,
            "status": "success",
            "confidence_score": 0.98,
        }

    typed_backend = _ConfigurableAgentBackend(sql_fn=mock_generate)
    monkeypatch.setattr("gemma_4_sql.sdk.agent.get_backend", _make_backend_getter(typed_backend))

    ctx = AgentContext(db_path=":memory:", ddl=ddl, max_retries=3)
    res = run_agentic_loop("google/gemma-4-2b", "Find users older than 30", backend=backend, context=ctx)
    assert isinstance(res, dict)
    assert res["backend"] == backend
    assert res["success"] is True
    assert res["attempts"] == 2
    assert res["final_sql"] == "SELECT name FROM users WHERE age > 30;"
    assert res["results"] == [("Alice",)]
    assert res["history"][0].get("success") is False
    assert res["history"][1].get("success") is True


def test_agentic_loop_invalid_backend() -> None:
    """Verify `run_agentic_loop` raises `ValueError` when an unknown backend is requested."""
    with pytest.raises(ValueError, match="Unknown backend: invalid"):
        run_agentic_loop("google/gemma-4-2b", "Get all users", backend="invalid")


def test_agentic_loop_self_correction_exhaustion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify agentic loop terminates with `success=False` when `max_retries` is exhausted."""
    ddl = "CREATE TABLE users (id INT, name TEXT);"

    typed_backend = _ConfigurableAgentBackend(
        static_res={
            "backend": "jax",
            "model": "m",
            "prompt": "p",
            "sql": "SELECT invalid_col FROM users;",
            "beam_width": 3,
            "confidence_score": 0.9,
            "status": "success",
        }
    )
    monkeypatch.setattr("gemma_4_sql.sdk.agent.get_backend", _make_backend_getter(typed_backend))

    ctx = AgentContext(db_path=":memory:", ddl=ddl, max_retries=2)
    res = run_agentic_loop("google/gemma-4-2b", "Get users", backend="jax", context=ctx)
    assert isinstance(res, dict)
    assert res["success"] is False
    assert res["attempts"] == 2
    assert len(res["history"]) == 2
    assert "no such column" in str(res["history"][0]["error"]).lower()


def test_agentic_loop_confidence_rejection_and_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify agentic loop rejects SQL below `min_confidence` and accepts SQL meeting `min_confidence`."""
    ddl = "CREATE TABLE users (id INT, name TEXT);"

    low_conf = _ConfigurableAgentBackend(
        static_res={
            "backend": "jax",
            "model": "m",
            "prompt": "p",
            "sql": "SELECT * FROM users;",
            "beam_width": 3,
            "status": "success",
            "confidence_score": 0.4,
        }
    )
    monkeypatch.setattr("gemma_4_sql.sdk.agent.get_backend", _make_backend_getter(low_conf))
    ctx = AgentContext(db_path=":memory:", ddl=ddl, max_retries=2, min_confidence=0.8)
    res = run_agentic_loop("google/gemma-4-2b", "Get users", backend="jax", context=ctx)
    assert isinstance(res, dict)
    assert res["success"] is False
    assert res["attempts"] == 2
    assert "below threshold" in str(res["history"][0]["error"])

    high_conf = _ConfigurableAgentBackend(
        static_res={
            "backend": "jax",
            "model": "m",
            "prompt": "p",
            "sql": "SELECT * FROM users;",
            "beam_width": 3,
            "status": "success",
            "confidence_score": 0.9,
        }
    )
    monkeypatch.setattr("gemma_4_sql.sdk.agent.get_backend", _make_backend_getter(high_conf))
    res_ok = run_agentic_loop("google/gemma-4-2b", "Get users", backend="jax", context=ctx)
    assert isinstance(res_ok, dict)
    assert res_ok["success"] is True
    assert res_ok["attempts"] == 1


@pytest.mark.asyncio
async def test_agentic_loop_async_and_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `run_agentic_loop_async` processes a batch of prompts concurrently via `asyncio.to_thread`."""
    ddl = "CREATE TABLE users (id INT, name TEXT);"
    typed_backend = _ConfigurableAgentBackend()
    assert typed_backend.generate_sql("m", "p")["sql"] == "SELECT * FROM jax_table;"
    typed_backend2 = _ConfigurableAgentBackend(
        static_res={
            "backend": "jax",
            "model": "m",
            "prompt": "p",
            "sql": "SELECT * FROM users;",
            "beam_width": 3,
            "status": "success",
            "confidence_score": 0.9,
        }
    )
    monkeypatch.setattr("gemma_4_sql.sdk.agent.get_backend", _make_backend_getter(typed_backend2))

    ctx = AgentContext(db_path=":memory:", ddl=ddl, max_retries=1)
    prompts = ["Get users", "Count users"]
    res = await run_agentic_loop_async("google/gemma-4-2b", prompts, backend="jax", context=ctx)
    assert isinstance(res, list)
    assert len(res) == 2
    assert res[0].get("success") is True
    assert res[1].get("success") is True


@pytest.mark.asyncio
async def test_process_single_prompt_error_truncation_and_hints(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `_process_single_prompt` truncates long errors and injects table/column hints."""
    typed_backend = _ConfigurableAgentBackend(
        static_res={
            "backend": "jax",
            "model": "m",
            "prompt": "p",
            "sql": "SELECT * FROM bad;",
            "beam_width": 3,
            "confidence_score": 1.0,
            "status": "success",
        }
    )

    from collections.abc import Iterator

    long_error = "no such table: bad " + "x" * 600
    empty_rows: list[tuple[JSONPrimitive, ...]] = []
    errors: Iterator[tuple[bool, list[tuple[JSONPrimitive, ...]], str | None]] = iter([
        (False, empty_rows, long_error),
        (False, empty_rows, "column 'foo' does not exist"),
        (False, empty_rows, "generic syntax error"),
    ])

    async def _scripted_feedback(_q: str) -> tuple[bool, list[tuple[JSONPrimitive, ...]], str | None]:
        """Return next scripted error feedback tuple."""
        return next(errors)

    async with LiveDatabaseEngine(":memory:") as engine:
        monkeypatch.setattr(engine, "execute_with_feedback_async", _scripted_feedback)
        ctx = AgentContext(max_retries=3)
        res = await _process_single_prompt("jax", typed_backend, "model", "prompt", engine, ctx)

    assert res.get("success") is False
    assert res["attempts"] == 3
    err0 = res["history"][0]["error"]
    assert err0 is not None and err0.endswith("... (truncated)")
    assert "[Hint: check available table names]" in str(res["history"][1].get("prompt"))
    assert "[Hint: check valid column names]" in str(res["history"][2].get("prompt"))


def test_agent_multimodal_and_running_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `AgentContext` multimodal parameters and sync-over-async running loop execution."""
    import gemma_4_sql.sdk.agent as ag

    typed_b = _ConfigurableAgentBackend(
        static_res={
            "backend": "pytorch",
            "model": "m",
            "prompt": "p",
            "sql": "SELECT 1",
            "beam_width": 3,
            "confidence_score": 0.9,
            "status": "ok",
        }
    )
    monkeypatch.setattr(ag, "get_backend", _make_backend_getter(typed_b))

    context = ag.AgentContext(
        ddl="CREATE TABLE t(id int)",
        image_path="fake.png",
        audio_path="fake.wav",
        multimodal_context={"emb": [1.0]},
    )

    async def _inside_running_loop() -> AgentResultDict | list[AgentResultDict]:
        """Call synchronous `run_agentic_loop` from inside an active asyncio event loop.

        Returns:
            Result from `run_agentic_loop`.

        """
        return ag.run_agentic_loop(prompt="test", context=context, model_name="m", backend="pytorch")

    res = asyncio.run(_inside_running_loop())
    assert isinstance(res, dict)
    assert res["status"] == "completed"
    assert typed_b.last_kwargs["image_path"] == "fake.png"
    assert typed_b.last_kwargs["audio_path"] == "fake.wav"
    assert typed_b.last_kwargs["multimodal_context"] == {"emb": [1.0]}

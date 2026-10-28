"""Behavioral and unit tests for the SDK Agent module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from gemma_4_sql.sdk.agent import (
    AgentContext,
    _process_single_prompt,
    classify_error_hint,
    extract_sql_from_response,
    run_agentic_loop,
    run_agentic_loop_async,
)
from gemma_4_sql.type_hints import AgentResultDict, InferenceResultDict, JSONValue


def test_extract_sql_from_response() -> None:
    """Verify extract_sql_from_response strips Markdown SQL fences and preserves raw SQL."""
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

    def mock_generate(model_name: str, current_prompt: str, **_kwargs: JSONValue) -> InferenceResultDict:
        """Simulate an initial invalid column query followed by a corrected valid SQL query."""
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

    mock_backend = MagicMock()
    mock_backend.generate_sql.side_effect = mock_generate
    monkeypatch.setattr("gemma_4_sql.sdk.agent.get_backend", lambda _b: mock_backend)  # pyright: ignore[reportUnknownLambdaType]

    ctx = AgentContext(db_path=":memory:", ddl=ddl, max_retries=3)
    res = run_agentic_loop("google/gemma-4-2b", "Find users older than 30", backend=backend, context=ctx)
    assert isinstance(res, dict)
    assert res["backend"] == backend
    assert res["success"] is True
    assert res["attempts"] == 2
    assert res["final_sql"] == "SELECT name FROM users WHERE age > 30;"
    assert res["results"] == [("Alice",)]
    assert res["history"][0]["success"] is False
    assert res["history"][1]["success"] is True


def test_agentic_loop_invalid_backend() -> None:
    """Verify run_agentic_loop raises ValueError when an unknown backend is requested."""
    with pytest.raises(ValueError, match="Unknown backend: invalid"):
        run_agentic_loop("google/gemma-4-2b", "Get all users", backend="invalid")


def test_agentic_loop_self_correction_exhaustion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify agentic loop terminates with success=False when max_retries is exhausted."""
    ddl = "CREATE TABLE users (id INT, name TEXT);"

    mock_backend = MagicMock()
    mock_backend.generate_sql.return_value = {"sql": "SELECT invalid_col FROM users;", "status": "success"}
    monkeypatch.setattr("gemma_4_sql.sdk.agent.get_backend", lambda _b: mock_backend)  # pyright: ignore[reportUnknownLambdaType]

    ctx = AgentContext(db_path=":memory:", ddl=ddl, max_retries=2)
    res = run_agentic_loop("google/gemma-4-2b", "Get users", backend="jax", context=ctx)
    assert isinstance(res, dict)
    assert res["success"] is False
    assert res["attempts"] == 2
    assert len(res["history"]) == 2
    assert "no such column" in str(res["history"][0]["error"]).lower()


def test_agentic_loop_confidence_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify agentic loop rejects SQL generations below min_confidence threshold."""
    ddl = "CREATE TABLE users (id INT, name TEXT);"

    mock_backend = MagicMock()
    mock_backend.generate_sql.return_value = {
        "sql": "SELECT * FROM users;",
        "status": "success",
        "confidence_score": 0.4,
    }
    monkeypatch.setattr("gemma_4_sql.sdk.agent.get_backend", lambda _b: mock_backend)  # pyright: ignore[reportUnknownLambdaType]

    ctx = AgentContext(db_path=":memory:", ddl=ddl, max_retries=2, min_confidence=0.8)
    res = run_agentic_loop("google/gemma-4-2b", "Get users", backend="jax", context=ctx)
    assert isinstance(res, dict)
    assert res["success"] is False
    assert res["attempts"] == 2
    assert "below threshold" in str(res["history"][0]["error"])


def test_agentic_loop_confidence_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify agentic loop accepts SQL generations meeting min_confidence threshold."""
    ddl = "CREATE TABLE users (id INT, name TEXT);"

    mock_backend = MagicMock()
    mock_backend.generate_sql.return_value = {
        "sql": "SELECT * FROM users;",
        "status": "success",
        "confidence_score": 0.9,
    }
    monkeypatch.setattr("gemma_4_sql.sdk.agent.get_backend", lambda _b: mock_backend)  # pyright: ignore[reportUnknownLambdaType]

    ctx = AgentContext(db_path=":memory:", ddl=ddl, max_retries=2, min_confidence=0.8)
    res = run_agentic_loop("google/gemma-4-2b", "Get users", backend="jax", context=ctx)
    assert isinstance(res, dict)
    assert res["success"] is True
    assert res["attempts"] == 1


@pytest.mark.asyncio
async def test_agentic_loop_async_and_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify run_agentic_loop_async processes a batch of prompts concurrently."""
    ddl = "CREATE TABLE users (id INT, name TEXT);"

    mock_backend = MagicMock()
    mock_backend.generate_sql.return_value = {
        "sql": "SELECT * FROM users;",
        "status": "success",
        "confidence_score": 0.9,
    }
    monkeypatch.setattr("gemma_4_sql.sdk.agent.get_backend", lambda _b: mock_backend)  # pyright: ignore[reportUnknownLambdaType]

    ctx = AgentContext(db_path=":memory:", ddl=ddl, max_retries=1)
    prompts = ["Get users", "Count users"]
    res = await run_agentic_loop_async("google/gemma-4-2b", prompts, backend="jax", context=ctx)
    assert isinstance(res, list)
    assert len(res) == 2
    assert res[0].get("success") is True
    assert res[1].get("success") is True


@pytest.mark.asyncio
async def test_process_single_prompt_error_truncation_and_hints() -> None:
    """Verify _process_single_prompt truncates long errors and injects table/column hints."""
    mock_backend = MagicMock()
    mock_backend.generate_sql.return_value = {"sql": "SELECT * FROM bad;", "confidence_score": 1.0}

    mock_engine = MagicMock()
    long_error = "no such table: bad " + "x" * 600
    mock_engine.execute_with_feedback_async = AsyncMock(
        side_effect=[
            (False, [], long_error),
            (False, [], "column 'foo' does not exist"),
            (False, [], "generic syntax error"),
        ]
    )

    ctx = AgentContext(max_retries=3)
    res = await _process_single_prompt("jax", mock_backend, "model", "prompt", mock_engine, ctx)

    assert res.get("success") is False
    assert res["attempts"] == 3
    err0 = res["history"][0]["error"]
    assert err0 is not None and err0.endswith("... (truncated)")
    assert "[Hint: check available table names]" in str(res["history"][1].get("prompt"))
    assert "[Hint: check valid column names]" in str(res["history"][2].get("prompt"))


def test_agent_multimodal_and_running_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify AgentContext multimodal parameters and sync-over-async running loop execution."""
    import asyncio

    import gemma_4_sql.sdk.agent as ag

    mock_b = MagicMock()
    mock_b.generate_sql.return_value = {"sql": "SELECT 1", "confidence_score": 0.9, "status": "ok"}
    monkeypatch.setattr(ag, "get_backend", lambda *_a: mock_b)  # pyright: ignore[reportUnknownLambdaType]

    context = ag.AgentContext(
        ddl="CREATE TABLE t(id int)",
        image_path="fake.png",
        audio_path="fake.wav",
        multimodal_context={"emb": [1.0]},
    )

    async def _inside_running_loop() -> AgentResultDict | list[AgentResultDict]:
        """Call synchronous run_agentic_loop from inside an active asyncio event loop."""
        return ag.run_agentic_loop(prompt="test", context=context, model_name="m", backend="pytorch")

    res = asyncio.run(_inside_running_loop())
    assert isinstance(res, dict)
    assert res["status"] == "completed"
    call_kwargs = mock_b.generate_sql.call_args.kwargs
    assert call_kwargs["image_path"] == "fake.png"
    assert call_kwargs["audio_path"] == "fake.wav"
    assert call_kwargs["multimodal_context"] == {"emb": [1.0]}

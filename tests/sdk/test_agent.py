"""Tests for SDK Agent module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from gemma_4_sql.sdk.agent import AgentContext, _process_single_prompt, run_agentic_loop
from gemma_4_sql.sdk.db_engine import LiveDatabaseEngine


def test_agentic_loop_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test run_agentic_loop with jax backend.

    Raises:
        AssertionError: Description.

    """
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    jax_agent = get_backend("jax")
    monkeypatch.setattr(jax_agent, "generate_sql", lambda *_args, **_kwargs: {"sql": "SELECT 1"})
    res = run_agentic_loop(model_name="model", prompt="prompt", backend="jax")
    assert res["backend"] == "jax"
    assert res["status"] == "completed"


def test_agentic_loop_keras(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test run_agentic_loop with keras backend.

    Raises:
        AssertionError: Description.

    """
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    keras_agent = get_backend("keras")
    monkeypatch.setattr(keras_agent, "generate_sql", lambda *_args, **_kwargs: {"sql": "SELECT 1"})
    res = run_agentic_loop(model_name="model", prompt="prompt", backend="keras")
    assert res["backend"] == "keras"
    assert res["status"] == "completed"


def test_agentic_loop_maxtext(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test run_agentic_loop with maxtext backend.

    Raises:
        AssertionError: Description.

    """
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    maxtext_agent = get_backend("maxtext")
    monkeypatch.setattr(maxtext_agent, "generate_sql", lambda *_args, **_kwargs: {"sql": "SELECT 1"})
    res = run_agentic_loop(model_name="model", prompt="prompt", backend="maxtext")
    assert res["backend"] == "maxtext"
    assert res["status"] == "completed"


def test_agentic_loop_pytorch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test run_agentic_loop with pytorch backend.

    Raises:
        AssertionError: Description.

    """
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    pytorch_agent = get_backend("pytorch")
    monkeypatch.setattr(pytorch_agent, "generate_sql", lambda *_args, **_kwargs: {"sql": "SELECT 1"})
    res = run_agentic_loop(model_name="model", prompt="prompt", backend="pytorch")
    assert res["backend"] == "pytorch"
    assert res["status"] == "completed"


def test_agentic_loop_invalid_backend() -> None:
    """Test run_agentic_loop with invalid backend."""
    with pytest.raises(ValueError, match="Unknown backend: invalid"):
        run_agentic_loop(model_name="model", prompt="prompt", backend="invalid")


@pytest.mark.asyncio
async def test_process_single_prompt_coverage() -> None:
    """Test process single prompt coverage."""
    backend_impl = MagicMock()
    backend_impl.generate_sql.side_effect = [
        {"sql": "SELECT 1", "confidence_score": 0.4},
        {"sql": "INVALID", "confidence_score": 0.9},
        {"sql": "INVALID", "confidence_score": 0.9},
        {"sql": "INVALID", "confidence_score": 0.9},
        {"sql": "SELECT 1", "confidence_score": 0.9},
        {"sql": "SELECT 1", "confidence_score": 0.9},
    ]
    engine = MagicMock()
    engine.execute_with_feedback_async = AsyncMock(
        side_effect=[
            (False, [], "syntax error near foo"),
            (False, [], "column foo does not exist"),
            (False, [], "no such table: bar"),
            (False, [], ""),
            (True, [(1,)], None),
        ]
    )
    ctx = AgentContext(min_confidence=0.5, max_retries=6)
    res = await _process_single_prompt("jax", backend_impl, "model", "prompt", engine, ctx)
    assert res["success"] is True


def test_agent_confidence_and_no_context(monkeypatch):
    """Test agent confidence and no context functionality."""

    class MockEngine(LiveDatabaseEngine):
        """Test class for MockEngine."""

        def __init__(self, **kwargs):
            """Initialize __init__."""

        def close(self):
            """Execute close helper."""

        async def execute_with_feedback_async(self, sql):
            """Execute execute with feedback async helper."""
            return (False, [], "err")

    class MockBackend:
        """Test class for MockBackend."""

        def generate_sql(self, m, p, **kwargs):
            """Execute generate sql helper."""
            return {"sql": "SELECT", "confidence_score": 0.5}

    import gemma_4_sql.sdk.agent as ag

    monkeypatch.setattr(ag, "get_backend", lambda x: MockBackend())

    monkeypatch.setattr(ag, "LiveDatabaseEngine", MockEngine)

    ctx = AgentContext(max_retries=1, min_confidence=0.8)
    res = ag.run_agentic_loop("model", "prompt", "jax", ctx)
    assert res["success"] is False
    assert len(res["history"]) == 1

    # And test no context
    res2 = ag.run_agentic_loop("model", "prompt", "jax", None)
    assert res2["success"] is False


def test_agent_success(monkeypatch):
    """Test agent success functionality."""
    import gemma_4_sql.sdk.agent as ag
    from gemma_4_sql.sdk.db_engine import LiveDatabaseEngine

    class MockEngine(LiveDatabaseEngine):
        """Test class for MockEngine."""

        def __init__(self, **kwargs):
            """Initialize __init__."""

        def close(self):
            """Execute close helper."""

        async def execute_with_feedback_async(self, sql):
            """Execute execute with feedback async helper."""
            return (True, [], None)

    class MockBackend:
        """Test class for MockBackend."""

        def generate_sql(self, m, p, **kwargs):
            """Execute generate sql helper."""
            return {"sql": "SELECT", "confidence_score": 0.9}

    monkeypatch.setattr(ag, "get_backend", lambda x: MockBackend())
    monkeypatch.setattr(ag, "LiveDatabaseEngine", MockEngine)

    ctx = ag.AgentContext(max_retries=1, min_confidence=0.8)
    res = ag.run_agentic_loop("model", "prompt", "jax", ctx)
    assert res["success"] is True


def test_agent_max_retries_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test agentic loop when max_retries is exhausted after continuous errors.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    import gemma_4_sql.sdk.agent as ag

    class AlwaysFailingEngine:
        """Engine that always fails."""

        def __init__(self, **kwargs: object) -> None:
            """Initialize."""

        def close(self) -> None:
            """Close."""

        async def execute_with_feedback_async(self, sql: str) -> tuple[bool, list, str]:
            """Return syntax error."""
            return (False, [], "persistent syntax error")

    class AlwaysSucceedingBackend:
        """Backend that generates high confidence SQL."""

        def generate_sql(self, m: str, p: str, **kwargs: object) -> dict:
            """Generate SQL."""
            return {"sql": "SELECT bad_col FROM table", "confidence_score": 0.99}

    monkeypatch.setattr(ag, "get_backend", lambda _: AlwaysSucceedingBackend())
    monkeypatch.setattr(ag, "LiveDatabaseEngine", AlwaysFailingEngine)

    ctx = ag.AgentContext(max_retries=3, min_confidence=0.5)
    res = ag.run_agentic_loop("model", "prompt", "jax", ctx)
    assert res["success"] is False
    assert res["attempts"] == 3
    assert len(res["history"]) == 3
    assert res["status"] == "completed"


@pytest.mark.asyncio
async def test_agent_from_running_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test run_agentic_loop when called from within an active running asyncio event loop.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    import gemma_4_sql.sdk.agent as ag

    class SimpleBackend:
        """Simple mock backend."""

        def generate_sql(self, m: str, p: str, **k: object) -> dict:
            """Generate simple SQL."""
            return {"sql": "SELECT 1", "confidence_score": 1.0}

    monkeypatch.setattr(ag, "get_backend", lambda _: SimpleBackend())
    res = ag.run_agentic_loop("m", "p", "jax", ag.AgentContext())
    assert "final_sql" in res


def test_agent_multimodal_context_and_table_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test agent passes multimodal context and formats table name hint on relation errors.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    import gemma_4_sql.sdk.agent as ag

    passed_kwargs: dict[str, object] = {}

    class MockBackend:
        """Mock backend capturing keyword arguments."""

        def generate_sql(self, m: str, p: str, **k: object) -> dict[str, object]:
            """Capture kwargs and return SQL."""
            passed_kwargs.update(k)
            # Second attempt succeeds
            if "Previous attempt failed" in p:
                assert "[Hint: check available table names]" in p
                return {"sql": "SELECT 1;", "confidence_score": 0.95}
            return {"sql": "SELECT * FROM missing_tbl;", "confidence_score": 0.9}

    class MockEngine:
        """Mock DB engine failing on missing table then succeeding."""

        async def execute_with_feedback_async(self, sql: str) -> tuple[bool, list[tuple[object, ...]], str | None]:
            """Return feedback."""
            if "missing_tbl" in sql:
                return (False, [], "Error: relation missing_tbl does not exist")
            return (True, [(1,)], None)

        def close(self) -> None:
            """Close engine."""

    monkeypatch.setattr(ag, "get_backend", lambda _: MockBackend())
    monkeypatch.setattr("gemma_4_sql.sdk.agent.LiveDatabaseEngine", lambda **_: MockEngine())

    context = ag.AgentContext(
        audio_path="/path/to/audio.wav",
        multimodal_context={"dummy": "embedding"},
        max_retries=2,
    )
    res = ag.run_agentic_loop("m", "p", "jax", context)
    assert res["success"] is True
    assert passed_kwargs["audio_path"] == "/path/to/audio.wav"
    assert passed_kwargs["multimodal_context"] == {"dummy": "embedding"}


def test_agent_with_image_path(monkeypatch):
    from unittest.mock import MagicMock

    import gemma_4_sql.sdk.agent as ag

    mock_b = MagicMock()
    mock_b.generate_sql.return_value = {"sql": "SELECT 1", "confidence_score": 1.0}
    monkeypatch.setattr(ag, "get_backend", lambda x: mock_b)
    """Test agent with image_path to cover missing line."""

    context = ag.AgentContext(ddl="CREATE TABLE t(id int)", image_path="fake.png")

    ag.run_agentic_loop(prompt="test", context=context, model_name="m", backend="pytorch")

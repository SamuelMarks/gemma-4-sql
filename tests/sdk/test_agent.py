"""Tests for SDK Agent module."""

import pytest

from gemma_4_sql.sdk.agent import run_agentic_loop


def test_agentic_loop_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test run_agentic_loop with jax backend.

    Raises:
        AssertionError: Description.

    """
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    jax_agent = get_backend("jax")
    monkeypatch.setattr(jax_agent, "generate_sql", lambda *_args, **_kwargs: {"sql": "SELECT 1"})
    res = run_agentic_loop(model_name="model", prompt="prompt", backend="jax")
    if res["backend"] != "jax":
        raise AssertionError
    if res["status"] != "completed":
        raise AssertionError


def test_agentic_loop_keras(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test run_agentic_loop with keras backend.

    Raises:
        AssertionError: Description.

    """
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    keras_agent = get_backend("keras")
    monkeypatch.setattr(keras_agent, "generate_sql", lambda *_args, **_kwargs: {"sql": "SELECT 1"})
    res = run_agentic_loop(model_name="model", prompt="prompt", backend="keras")
    if res["backend"] != "keras":
        raise AssertionError
    if res["status"] != "completed":
        raise AssertionError


def test_agentic_loop_maxtext(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test run_agentic_loop with maxtext backend.

    Raises:
        AssertionError: Description.

    """
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    maxtext_agent = get_backend("maxtext")
    monkeypatch.setattr(maxtext_agent, "generate_sql", lambda *_args, **_kwargs: {"sql": "SELECT 1"})
    res = run_agentic_loop(model_name="model", prompt="prompt", backend="maxtext")
    if res["backend"] != "maxtext":
        raise AssertionError
    if res["status"] != "completed":
        raise AssertionError


def test_agentic_loop_pytorch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test run_agentic_loop with pytorch backend.

    Raises:
        AssertionError: Description.

    """
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    pytorch_agent = get_backend("pytorch")
    monkeypatch.setattr(pytorch_agent, "generate_sql", lambda *_args, **_kwargs: {"sql": "SELECT 1"})
    res = run_agentic_loop(model_name="model", prompt="prompt", backend="pytorch")
    if res["backend"] != "pytorch":
        raise AssertionError
    if res["status"] != "completed":
        raise AssertionError


def test_agentic_loop_invalid_backend() -> None:
    """Test run_agentic_loop with invalid backend."""
    with pytest.raises(ValueError, match="Unknown backend: invalid"):
        run_agentic_loop(model_name="model", prompt="prompt", backend="invalid")


from unittest.mock import AsyncMock, MagicMock

import pytest

from gemma_4_sql.sdk.agent import AgentContext, _process_single_prompt
from gemma_4_sql.sdk.db_engine import LiveDatabaseEngine


@pytest.mark.asyncio
async def test_process_single_prompt_coverage() -> None:
    """Test process single prompt coverage."""
    backend_impl = MagicMock()
    backend_impl.generate_sql.side_effect = [{"sql": "SELECT 1", "confidence_score": 0.4}, {"sql": "INVALID", "confidence_score": 0.9}, {"sql": "SELECT 1", "confidence_score": 0.9}]
    engine = MagicMock()
    engine.execute_with_feedback_async = AsyncMock(side_effect=[(False, [], "syntax error"), (True, [(1,)], None)])
    ctx = AgentContext(min_confidence=0.5)
    res = await _process_single_prompt("jax", backend_impl, "model", "prompt", engine, ctx)
    assert res["success"] is True


def test_agent_confidence_and_no_context(monkeypatch):
    class MockEngine(LiveDatabaseEngine):
        def __init__(self, **kwargs):
            pass

        def close(self):
            pass

        async def execute_with_feedback_async(self, sql):
            return (False, [], "err")

    class MockBackend:
        def generate_sql(self, m, p):
            return {"sql": "SELECT", "confidence_score": 0.5}

    import sys

    monkeypatch.setitem(sys.modules, "gemma_4_sql.sdk.registry", type("Reg", (), {"get_backend": lambda x: MockBackend()}))
    import builtins

    orig_import = builtins.__import__

    def mock_import(name, *a, **k):
        if name == "gemma_4_sql.sdk.registry":
            return sys.modules["gemma_4_sql.sdk.registry"]
        return orig_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    import gemma_4_sql.sdk.agent as ag

    monkeypatch.setattr(ag, "LiveDatabaseEngine", MockEngine)

    ctx = AgentContext(max_retries=1, min_confidence=0.8)
    res = ag.run_agentic_loop("model", "prompt", "jax", ctx)
    assert res["success"] is False
    assert len(res["history"]) == 1

    # And test no context
    res2 = ag.run_agentic_loop("model", "prompt", "jax", None)
    assert res2["success"] is False


def test_agent_success(monkeypatch):
    import gemma_4_sql.sdk.agent as ag
    from gemma_4_sql.sdk.db_engine import LiveDatabaseEngine

    class MockEngine(LiveDatabaseEngine):
        def __init__(self, **kwargs):
            pass

        def close(self):
            pass

        async def execute_with_feedback_async(self, sql):
            return (True, [], None)

    class MockBackend:
        def generate_sql(self, m, p):
            return {"sql": "SELECT", "confidence_score": 0.9}

    import sys

    monkeypatch.setitem(sys.modules, "gemma_4_sql.sdk.registry", type("Reg", (), {"get_backend": lambda x: MockBackend()}))
    import builtins

    orig_import = builtins.__import__

    def mock_import(name, *a, **k):
        if name == "gemma_4_sql.sdk.registry":
            return sys.modules["gemma_4_sql.sdk.registry"]
        return orig_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    monkeypatch.setattr(ag, "LiveDatabaseEngine", MockEngine)

    ctx = ag.AgentContext(max_retries=1, min_confidence=0.8)
    res = ag.run_agentic_loop("model", "prompt", "jax", ctx)
    assert res["success"] is True

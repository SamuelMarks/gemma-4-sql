"""Tests for SDK Chat module."""

from __future__ import annotations

from typing import NoReturn as Never

import pytest

from gemma_4_sql.sdk.chat import chat_turn


def test_chat_turn_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test chat_turn with multiple backends.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Raises:
        AssertionError: If backend routing assertion fails.
    """
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    for backend in ["jax", "keras", "maxtext", "pytorch"]:
        backend_impl = get_backend(backend)
        monkeypatch.setattr(backend_impl, "generate_sql", lambda *_args, **_kwargs: {"sql": "SELECT 1"})
        res = chat_turn("foo", [{"role": "user", "content": "hi"}], "prompt", backend=backend)
        if res["backend"] != backend:
            raise AssertionError
        if res["model"] != "foo":
            raise AssertionError
        if res["response"] != "SELECT 1":
            raise AssertionError
        if len(res["history"]) != int("3"):
            raise AssertionError


def test_chat_turn_routing_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test chat_turn error handling.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Raises:
        AssertionError: Description.
    """
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    backend_impl = get_backend("jax")

    def mock_generate(*_args: object, **_kwargs: object) -> Never:
        """Execute function.

        Raises:
            ValueError: Description.
        """
        msg = "mock error"
        raise ValueError(msg)

    monkeypatch.setattr(backend_impl, "generate_sql", mock_generate)
    with pytest.raises(RuntimeError, match="Chat turn failed"):
        chat_turn("foo", [], "prompt", backend="jax")


def test_multiturn_history_accumulation_and_prompt_formatting(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test history accumulation and system/user/assistant prompt formatting.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    captured_prompts: list[str] = []

    def mock_generate(_model: str, prompt: str, **_kwargs: object) -> dict[str, str]:
        """Capture generated prompt and return mock SQL.

        Args:
            _model: Model name.
            prompt: Prompt string.
            **_kwargs: Keyword arguments.

        Returns:
            Dictionary with SQL response.
        """
        captured_prompts.append(prompt)
        return {"sql": f"SELECT * FROM tbl_{len(captured_prompts)}"}

    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    backend_impl = get_backend("jax")
    monkeypatch.setattr(backend_impl, "generate_sql", mock_generate)

    initial_history = [
        {"role": "system", "content": "You are a SQL assistant."},
        {"role": "user", "content": "Show me employees."},
        {"role": "assistant", "content": "SELECT * FROM employees;"},
    ]

    # Turn 1
    res1 = chat_turn("model_v1", initial_history, "Filter by department Sales", backend="jax")
    assert res1["response"] == "SELECT * FROM tbl_1"
    assert len(res1["history"]) == 5
    assert res1["history"][-2] == {"role": "user", "content": "Filter by department Sales"}
    assert res1["history"][-1] == {"role": "assistant", "content": "SELECT * FROM tbl_1"}
    assert "system: You are a SQL assistant.\n" in captured_prompts[0]
    assert "user: Show me employees.\n" in captured_prompts[0]
    assert "assistant: SELECT * FROM employees;\n" in captured_prompts[0]
    assert "user: Filter by department Sales\nassistant: " in captured_prompts[0]

    # Turn 2
    res2 = chat_turn("model_v1", res1["history"], "Sort by salary descending", backend="jax")
    assert res2["response"] == "SELECT * FROM tbl_2"
    assert len(res2["history"]) == 7
    assert res2["history"][-2] == {"role": "user", "content": "Sort by salary descending"}
    assert res2["history"][-1] == {"role": "assistant", "content": "SELECT * FROM tbl_2"}
    assert "user: Filter by department Sales\n" in captured_prompts[1]
    assert "assistant: SELECT * FROM tbl_1\n" in captured_prompts[1]
    assert "user: Sort by salary descending\nassistant: " in captured_prompts[1]


def test_chat_turn_missing_sql_response_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test error when backend response does not contain SQL key.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    backend_impl = get_backend("jax")

    monkeypatch.setattr(backend_impl, "generate_sql", lambda *_a, **_k: {"status": "error_no_sql"})
    with pytest.raises(RuntimeError, match=r"Chat turn failed: Backend jax did not return SQL\."):
        chat_turn("model_v1", [], "test prompt", backend="jax")

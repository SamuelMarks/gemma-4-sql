"""Tests for SDK Chat module."""

from __future__ import annotations

from typing import NoReturn as Never

import pytest

from gemma_4_sql.sdk.chat import chat_turn
from gemma_4_sql.sdk.registry import get_backend
from gemma_4_sql.type_hints import JSONValue


def test_chat_turn_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test chat_turn with multiple backends.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Raises:
        AssertionError: If backend routing assertion fails.

    """
    for backend in ["jax", "keras", "maxtext", "pytorch"]:
        backend_impl = get_backend(backend)
        monkeypatch.setattr(backend_impl, "generate_sql", lambda *_args, **_kwargs: {"sql": "SELECT 1"})  # pyright: ignore[reportUnknownLambdaType]
        res = chat_turn("foo", [{"role": "user", "content": "hi"}], "prompt", backend=backend)
        assert res["backend"] == backend
        assert res["model"] == "foo"
        assert res["response"] == "SELECT 1"
        assert len(res["history"]) == 3


def test_chat_turn_routing_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify chat turn routing error behavior."""
    backend_impl = get_backend("jax")

    def mock_generate(*_args: str | int, **_kwargs: JSONValue) -> Never:
        """Provide mock generate test helper."""
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

    def mock_generate(_model: str, prompt: str, **_kwargs: JSONValue) -> dict[str, str]:
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

    backend_impl = get_backend("jax")
    monkeypatch.setattr(backend_impl, "generate_sql", mock_generate)

    initial_history = [
        {"role": "system", "content": "You are a SQL assistant."},
        {"role": "user", "content": "Show me employees."},
        {"role": "assistant", "content": "SELECT * FROM employees;"},
    ]

    nl = chr(10)
    # Turn 1
    res1 = chat_turn("model_v1", initial_history, "Filter by department Sales", backend="jax")
    assert res1["response"] == "SELECT * FROM tbl_1"
    assert len(res1["history"]) == 5
    assert res1["history"][-2] == {"role": "user", "content": "Filter by department Sales"}
    assert res1["history"][-1] == {"role": "assistant", "content": "SELECT * FROM tbl_1"}
    assert f"system: You are a SQL assistant.{nl}" in captured_prompts[0]
    assert f"user: Show me employees.{nl}" in captured_prompts[0]
    assert f"assistant: SELECT * FROM employees;{nl}" in captured_prompts[0]
    assert f"user: Filter by department Sales{nl}assistant: " in captured_prompts[0]

    # Turn 2
    res2 = chat_turn("model_v1", res1["history"], "Sort by salary descending", backend="jax")
    assert res2["response"] == "SELECT * FROM tbl_2"
    assert len(res2["history"]) == 7
    assert res2["history"][-2] == {"role": "user", "content": "Sort by salary descending"}
    assert res2["history"][-1] == {"role": "assistant", "content": "SELECT * FROM tbl_2"}
    assert f"user: Filter by department Sales{nl}" in captured_prompts[1]
    assert f"assistant: SELECT * FROM tbl_1{nl}" in captured_prompts[1]
    assert f"user: Sort by salary descending{nl}assistant: " in captured_prompts[1]


def test_chat_turn_missing_sql_response_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test error when backend response does not contain SQL key.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    """
    backend_impl = get_backend("jax")

    monkeypatch.setattr(backend_impl, "generate_sql", lambda *_a, **_k: {"status": "error_no_sql"})  # pyright: ignore[reportUnknownLambdaType]
    with pytest.raises(RuntimeError, match=r"Chat turn failed: Backend jax did not return SQL\."):
        chat_turn("model_v1", [], "test prompt", backend="jax")

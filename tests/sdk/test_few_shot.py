"""Tests for few-shot dynamic prompt generation and selection."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gemma_4_sql.sdk.few_shot import (
    build_few_shot_prompt,
    generate_few_shot_sql,
    select_relevant_examples,
)


def test_few_shot_routing() -> None:
    """Test few-shot prompt routing across all backends."""
    for backend in ["jax", "keras", "maxtext", "pytorch", "unknown"]:
        res = build_few_shot_prompt("foo", "prompt", [{"input": "a", "output": "b"}], backend=backend)
        assert res["backend"] == backend
        assert res["model"] == "foo"
        assert res["status"] == f"success_{backend}_few_shot"
        assert "Input: a\nOutput: b" in str(res["few_shot_prompt"])


def test_select_relevant_examples() -> None:
    """Test dynamic selection of few-shot examples with budgeting."""
    empty = select_relevant_examples("Find users", [])
    assert empty == []

    pool = [
        {"input": "Count all active users in department", "output": "SELECT COUNT(*) FROM users WHERE active = 1"},
        {"input": "Show products with low stock", "output": "SELECT * FROM products WHERE stock < 10"},
        {"input": "List users created this month", "output": "SELECT * FROM users WHERE created_at >= '2026-09-01'"},
    ]

    selected = select_relevant_examples("How many users exist?", pool, top_k=2)
    assert len(selected) == 2
    assert "users" in selected[0]["input"].lower()

    # Test budgeting truncation
    limited = select_relevant_examples("users", pool, top_k=3, max_tokens=10)
    assert len(limited) <= 2


def test_build_few_shot_prompt_with_selection() -> None:
    """Test build_few_shot_prompt with top_k selection."""
    pool = [
        {"input": "Query customers", "output": "SELECT * FROM customers"},
        {"input": "Query orders", "output": "SELECT * FROM orders"},
    ]
    res = build_few_shot_prompt("test_model", "Find customers", pool, backend="jax", top_k=1)
    assert res["num_examples"] == 1
    assert "customers" in str(res["few_shot_prompt"])
    assert "orders" not in str(res["few_shot_prompt"])

    # Empty examples
    res_empty = build_few_shot_prompt("test_model", "Find customers", [], backend="jax")
    assert res_empty["few_shot_prompt"] == "Input: Find customers\nOutput: "


def test_generate_few_shot_sql(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test end-to-end generate_few_shot_sql using registry backend."""
    mock_backend = MagicMock()
    mock_backend.generate_sql.return_value = {
        "sql": "SELECT COUNT(*) FROM users",
        "confidence_score": 0.92,
        "status": "success",
    }
    monkeypatch.setattr("gemma_4_sql.sdk.registry.get_backend", lambda _name: mock_backend)

    pool = [{"input": "Count users", "output": "SELECT count(*) FROM users"}]
    res = generate_few_shot_sql(
        model_name="gemma-4",
        prompt="How many users?",
        examples=pool,
        backend="jax",
        top_k=1,
    )
    assert res["status"] == "success"
    assert res["sql"] == "SELECT COUNT(*) FROM users"
    assert res["confidence_score"] == 0.92
    assert res["num_examples"] == 1

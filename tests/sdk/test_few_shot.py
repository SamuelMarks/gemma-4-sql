"""Tests for few-shot dynamic prompt generation, multimodal placeholders, and SQL generation."""

from __future__ import annotations

import pytest

from gemma_4_sql.exceptions import ConfigurationError
from gemma_4_sql.sdk.few_shot import (
    build_few_shot_prompt,
    generate_few_shot_sql,
    select_relevant_examples,
)
from tests.utils.test_utils import RecordingBackendDouble


def test_select_relevant_examples() -> None:
    """Verify `select_relevant_examples` scores keyword overlap and enforces `max_tokens` budgets."""
    assert select_relevant_examples("count users", []) == []

    pool = [
        {"input": "List all orders", "output": "SELECT * FROM orders"},
        {"input": "Count active users", "output": "SELECT COUNT(*) FROM users WHERE active = 1"},
        {"input": "Get product prices", "output": "SELECT price FROM products"},
    ]

    selected = select_relevant_examples("Count total users", pool, top_k=2)
    assert len(selected) == 2
    assert selected[0]["input"] == "Count active users"

    # Tight token budget prunes subsequent examples after the 81-char top-ranked match
    selected_tight = select_relevant_examples("Count total users", pool, top_k=3, max_tokens=22)
    assert len(selected_tight) == 1
    assert selected_tight[0]["input"] == "Count active users"


def test_build_few_shot_prompt_and_multimodal_flags() -> None:
    """Verify `build_few_shot_prompt` handles `top_k`, budgets, multimodal flags, and unknown backends."""
    pool = [
        {"input": "Find customers", "output": "SELECT * FROM customers"},
        {"input": "Query orders", "output": "SELECT * FROM orders"},
    ]
    res = build_few_shot_prompt("test_model", "Find customers", pool, backend="jax", top_k=1)
    assert res["num_examples"] == 1
    assert "customers" in str(res["few_shot_prompt"])
    assert "orders" not in str(res["few_shot_prompt"])

    res_empty = build_few_shot_prompt("test_model", "Find customers", [], backend="jax")
    nl = """
"""
    assert res_empty["few_shot_prompt"] == f"Input: Find customers{nl}Output: "

    res_budget = build_few_shot_prompt("test_model", "Find customers", pool, backend="jax", top_k=None, max_tokens=15)
    assert res_budget["num_examples"] == 1

    # Multimodal placeholders (`image_path` / `audio_path` / `modality`)
    res_mm = build_few_shot_prompt(
        "test_model",
        "Find customers",
        pool,
        backend="jax",
        top_k=1,
        image_path="erd.png",
        audio_path="q.wav",
        modality="multimodal",
    )
    assert "<image>" in str(res_mm["few_shot_prompt"])
    assert "<audio>" in str(res_mm["few_shot_prompt"])

    with pytest.raises(ConfigurationError, match="Unknown backend: invalid_be"):
        build_few_shot_prompt("test_model", "Find customers", pool, backend="invalid_be")


def test_generate_few_shot_sql(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify end-to-end `generate_few_shot_sql` using a typed `RecordingBackendDouble`."""
    fake_backend = RecordingBackendDouble(backend_name="jax")

    def _get_fake_backend(_name: str) -> RecordingBackendDouble:
        """Return `fake_backend`."""
        return fake_backend

    monkeypatch.setattr("gemma_4_sql.sdk.few_shot.get_backend", _get_fake_backend)

    pool = [{"input": "Count users", "output": "SELECT count(*) FROM users"}]
    res = generate_few_shot_sql(
        model_name="gemma-4",
        prompt="How many users?",
        examples=pool,
        backend="jax",
        top_k=1,
        image_path="erd.png",
        modality="vision",
    )
    assert res["status"] == "success"
    assert res.get("sql") == "SELECT * FROM jax_table;"
    assert res.get("confidence_score") == 0.95
    assert res.get("num_examples") == 1
    assert "<image>" in str(res.get("few_shot_prompt"))

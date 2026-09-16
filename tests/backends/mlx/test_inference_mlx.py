"""Tests for MLX inference and beam search."""

from __future__ import annotations

import mlx.core as mx
import pytest

import gemma_4_sql.backends.mlx.inference as mlx_inf
from gemma_4_sql.backends.mlx.inference import (
    compute_confidence_score,
    generate_sql,
    mlx_beam_search,
)
from gemma_4_sql.exceptions import DependencyMissingError, InferenceError


def test_compute_confidence_score() -> None:
    """Test confidence score mathematical properties and bounds."""
    # Zero tokens boundary
    assert compute_confidence_score([], 0) == 0.0

    # Perfect confidence (0 log prob)
    assert compute_confidence_score(0.0, 5) == pytest.approx(1.0)
    assert compute_confidence_score([0.0, 0.0, 0.0], 3) == pytest.approx(1.0)

    # High confidence: log(0.9) approx -0.10536
    score = compute_confidence_score([-0.10536, -0.10536], 2)
    assert score == pytest.approx(0.9, rel=1e-2)

    # Positive log prob overflow guard
    assert compute_confidence_score(10.0, 2) == 1.0


def test_mlx_beam_search_success_with_mx() -> None:
    """Test beam search with MLX arrays producing valid SQL."""

    class MockTokenizer:
        eos_token_id = 99

        def encode(self, prompt: str) -> list[int]:
            return [10, 20]

        def decode(self, tokens: list[int]) -> str:
            return "SELECT * FROM users WHERE id = 1"

    # Mock model: returns 3D logits of shape (1, seq_len, vocab_size=100)
    step = 0

    def mock_model(inputs: mx.array) -> mx.array:
        nonlocal step
        step += 1
        seq_len = inputs.shape[-1]
        logits = mx.zeros((1, seq_len, 100))
        # Top token changes per step until EOS
        target_token = 99 if step >= 3 else 42
        logits[0, -1, target_token] = 10.0
        return logits

    sql, confidence = mlx_beam_search(
        model=mock_model,
        tokenizer=MockTokenizer(),
        prompt="Find user 1",
        beam_width=2,
        max_length=5,
    )
    assert sql == "SELECT * FROM users WHERE id = 1"
    assert 0.0 < confidence <= 1.0


def test_mlx_beam_search_greedy_width_1() -> None:
    """Test greedy decoding when beam_width is 1."""

    class MockTokenizer:
        eos_token_id = 99

        def encode(self, _prompt: str) -> list[int]:
            return [1]

        def decode(self, tokens: list[int]) -> str:
            return "SELECT 1"

    step = 0

    def mock_model(inputs: mx.array) -> mx.array:
        nonlocal step
        step += 1
        seq_len = inputs.shape[-1]
        logits = mx.zeros((seq_len, 100))
        target_token = 99 if step >= 2 else 5
        logits[-1, target_token] = 5.0
        return logits

    sql, confidence = mlx_beam_search(
        model=mock_model,
        tokenizer=MockTokenizer(),
        prompt="Query",
        beam_width=1,
        max_length=4,
    )
    assert sql == "SELECT 1"
    assert confidence > 0.0


def test_mlx_beam_search_python_list_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test beam search when model returns python list structures."""
    monkeypatch.setattr(mlx_inf, "mx", None)

    step = 0

    def mock_model(_inputs: list[int]) -> list[list[float]]:
        nonlocal step
        step += 1
        logits = [0.1] * 10
        target = 1 if step >= 2 else 3
        logits[target] = 5.0
        return [logits]

    sql, conf = mlx_beam_search(
        model=mock_model,
        tokenizer=None,
        prompt="Get all",
        beam_width=2,
        max_length=3,
        eos_token_id=1,
    )
    assert isinstance(sql, str)
    assert len(sql) > 0
    assert 0.0 < conf <= 1.0


def test_mlx_beam_search_errors() -> None:
    """Test error conditions in mlx_beam_search."""
    # None model
    with pytest.raises(InferenceError, match="Valid model instance is required"):
        mlx_beam_search(model=None, tokenizer=None, prompt="test")

    # Empty sequence
    def empty_model(_inputs: object) -> list[list[float]]:
        return []

    with pytest.raises(InferenceError, match="MLX beam search yielded an empty sequence"):
        mlx_beam_search(model=empty_model, tokenizer=None, prompt="test", max_length=0)

    # Empty SQL query string
    class EmptyTokenizer:
        eos_token_id = 99

        def encode(self, _p: str) -> list[int]:
            return [1]

        def decode(self, _t: list[int]) -> str:
            return "   "

    def dummy_model(_inputs: mx.array) -> mx.array:
        return mx.ones((1, 1, 100))

    with pytest.raises(InferenceError, match="decoded into an empty SQL query string"):
        mlx_beam_search(model=dummy_model, tokenizer=EmptyTokenizer(), prompt="test", max_length=1)


def test_generate_sql_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test generate_sql top-level wrapper with successful decoding."""

    class MockTokenizer:
        eos_token_id = 99

        def encode(self, prompt: str) -> list[int]:
            return [10]

        def decode(self, tokens: list[int]) -> str:
            return "SELECT COUNT(*) FROM orders"

    step = 0

    def mock_model(inputs: mx.array) -> mx.array:
        nonlocal step
        step += 1
        seq_len = inputs.shape[-1]
        logits = mx.zeros((1, seq_len, 100))
        # First step generates token 42, next step generates EOS (99)
        tok = 99 if step >= 2 else 42
        logits[0, -1, tok] = 10.0
        return logits

    monkeypatch.setattr(mlx_inf, "load", lambda _name: (mock_model, MockTokenizer()))

    res = generate_sql("mock_mlx_model", "Count orders", beam_width=3, max_length=10, eos_token_id=99)
    assert res["backend"] == "mlx"
    assert res["model"] == "mock_mlx_model"
    assert res["sql"] == "SELECT COUNT(*) FROM orders"
    assert res["status"] == "success"
    assert res["confidence_score"] > 0.0


def test_generate_sql_missing_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test generate_sql raises DependencyMissingError when load is missing."""
    monkeypatch.setattr(mlx_inf, "load", None)
    with pytest.raises(DependencyMissingError, match="MLX dependencies are missing"):
        generate_sql("model", "prompt")


def test_generate_sql_failure_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test generate_sql gracefully catches inference failure."""

    def fail_load(_name: str) -> object:
        raise RuntimeError("Model checkpoint corrupted")

    monkeypatch.setattr(mlx_inf, "load", fail_load)
    res = generate_sql("bad_model", "prompt")
    assert res["status"] == "failed: Model checkpoint corrupted"
    assert res["sql"] == ""
    assert res["confidence_score"] == 0.0


def test_mlx_beam_search_only_eos_error() -> None:
    """Test InferenceError when beam search produces only an EOS token."""

    class EosTokenizer:
        eos_token_id = 99

        def encode(self, _p: str) -> list[int]:
            return [1]

        def decode(self, _t: list[int]) -> str:
            return "SELECT 1"

    # Immediately emits EOS token as first token
    def eos_model(_inputs: mx.array) -> mx.array:
        logits = mx.zeros((1, 1, 100))
        logits[0, -1, 99] = 10.0
        return logits

    with pytest.raises(InferenceError, match="yielded only an EOS token"):
        mlx_beam_search(model=eos_model, tokenizer=EosTokenizer(), prompt="test", max_length=2)


def test_mlx_beam_search_1d_logits_and_single_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test 1D logits and single model without tokenizer in load."""

    def mock_model(inputs: mx.array) -> mx.array:
        # Returns 1D logits with ASCII for 'S' (83)
        logits = mx.zeros((100,))
        logits[83] = 5.0
        return logits

    monkeypatch.setattr(mlx_inf, "load", lambda _name: mock_model)
    # Also test early stopping when step hits max
    res = generate_sql("mock_single_model", "test query", beam_width=1, max_length=2)
    assert res["status"] == "success"
    assert "S" in res["sql"]


def test_mlx_beam_search_early_done_break() -> None:
    """Test early loop termination when all beams hit is_done."""

    class MockTokenizer:
        eos_token_id = 99

        def encode(self, _p: str) -> list[int]:
            return [1]

        def decode(self, _t: list[int]) -> str:
            return "SELECT id FROM t"

    step = 0

    def mock_model(_inputs: mx.array) -> mx.array:
        nonlocal step
        step += 1
        logits = mx.zeros((1, 1, 100))
        # Always output EOS token on first step
        logits[0, -1, 99] = 10.0
        return logits

    # Output tokens will include 42 then 99
    step_cnt = 0

    def mock_model_2(_inputs: mx.array) -> mx.array:
        nonlocal step_cnt
        step_cnt += 1
        logits = mx.zeros((1, 1, 100))
        if step_cnt == 1:
            logits[0, -1, 42] = 10.0
        else:
            logits[0, -1, 99] = 10.0
        return logits

    sql, conf = mlx_beam_search(
        model=mock_model_2,
        tokenizer=MockTokenizer(),
        prompt="Query",
        beam_width=1,
        max_length=10,
    )
    assert sql == "SELECT id FROM t"
    assert conf > 0.0


def test_mlx_beam_search_unrecognized_logits_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test fallback when logits output is not array or list."""
    monkeypatch.setattr(mlx_inf, "mx", None)

    def mock_model(_inputs: list[int]) -> object:
        return "unrecognized_output"

    with pytest.raises(InferenceError, match="yielded only an EOS token"):
        mlx_beam_search(
            model=mock_model,
            tokenizer=None,
            prompt="Query",
            beam_width=1,
            max_length=2,
            eos_token_id=1,
        )

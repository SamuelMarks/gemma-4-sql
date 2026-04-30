"""
Tests for SDK Evaluation module.
"""

import pytest

from gemma_4_sql.sdk.evaluation import evaluate


def test_evaluate_jax() -> None:
    """Test evaluate with jax."""
    res = evaluate("model1", "data1", "jax")
    assert res["backend"] == "jax"
    assert res["model"] == "model1"
    assert res["dataset"] == "data1"
    assert res["status"] == "completed"
    assert "metrics" in res


def test_evaluate_keras() -> None:
    """Test evaluate with keras."""
    res = evaluate("model1", "data1", "keras")
    assert res["backend"] == "keras"
    assert res["model"] == "model1"
    assert res["dataset"] == "data1"
    assert res["status"] == "completed"
    assert "metrics" in res


def test_evaluate_maxtext() -> None:
    """Test evaluate with maxtext."""
    res = evaluate("model1", "data1", "maxtext")
    assert res["backend"] == "maxtext"
    assert res["model"] == "model1"
    assert res["dataset"] == "data1"
    assert res["status"] == "completed"
    assert "metrics" in res


def test_evaluate_pytorch() -> None:
    """Test evaluate with pytorch."""
    res = evaluate("model1", "data1", "pytorch")
    assert res["backend"] == "pytorch"
    assert res["model"] == "model1"
    assert res["dataset"] == "data1"
    assert res["status"] == "completed"
    assert "metrics" in res


def test_evaluate_invalid() -> None:
    """Test evaluate with invalid backend."""
    with pytest.raises(ValueError, match="Unknown backend: invalid"):
        evaluate("model1", "data1", "invalid")

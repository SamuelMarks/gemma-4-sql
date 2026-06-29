"""Tests for SDK Evaluation module."""

import pytest
from gemma_4_sql.sdk.evaluation import evaluate


def test_evaluate_jax() -> None:
    """Test evaluate with jax."""
    res = evaluate("model1", "data1", "jax")
    if not res["backend"] == "jax":
        raise AssertionError
    if not res["model"] == "model1":
        raise AssertionError
    if not res["dataset"] == "data1":
        raise AssertionError
    if not res["status"] == "completed":
        raise AssertionError
    if "metrics" not in res:
        raise AssertionError


def test_evaluate_keras() -> None:
    """Test evaluate with keras."""
    res = evaluate("model1", "data1", "keras")
    if not res["backend"] == "keras":
        raise AssertionError
    if not res["model"] == "model1":
        raise AssertionError
    if not res["dataset"] == "data1":
        raise AssertionError
    if not res["status"] == "completed":
        raise AssertionError
    if "metrics" not in res:
        raise AssertionError


def test_evaluate_maxtext() -> None:
    """Test evaluate with maxtext."""
    res = evaluate("model1", "data1", "maxtext")
    if not res["backend"] == "maxtext":
        raise AssertionError
    if not res["model"] == "model1":
        raise AssertionError
    if not res["dataset"] == "data1":
        raise AssertionError
    if not res["status"] == "completed":
        raise AssertionError
    if "metrics" not in res:
        raise AssertionError


def test_evaluate_pytorch() -> None:
    """Test evaluate with pytorch."""
    res = evaluate("model1", "data1", "pytorch")
    if not res["backend"] == "pytorch":
        raise AssertionError
    if not res["model"] == "model1":
        raise AssertionError
    if not res["dataset"] == "data1":
        raise AssertionError
    if not res["status"] == "completed":
        raise AssertionError
    if "metrics" not in res:
        raise AssertionError


def test_evaluate_invalid() -> None:
    """Test evaluate with invalid backend."""
    with pytest.raises(ValueError, match="Unknown backend: invalid"):
        evaluate("model1", "data1", "invalid")

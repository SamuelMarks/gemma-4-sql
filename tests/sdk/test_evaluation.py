"""Tests for SDK Evaluation module."""

import pytest

from gemma_4_sql.sdk.evaluation import evaluate


def test_evaluate_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test evaluate with jax."""
    monkeypatch.setattr("gemma_4_sql.backends.jax.evaluate.generate_sql", lambda *_args, **_kwargs: {"sql": "SELECT 1"})
    res = evaluate("model1", "data1", "jax")
    if not res["backend"] == "jax":
        raise AssertionError
    if not res["model"] == "model1":
        raise AssertionError
    if not res["dataset"] == "data1":
        raise AssertionError
    if "metrics" not in res:
        raise AssertionError


def test_evaluate_keras(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test evaluate with keras."""
    monkeypatch.setattr("gemma_4_sql.backends.keras.evaluate.generate_sql", lambda *_args, **_kwargs: {"sql": "SELECT 1"})
    res = evaluate("model1", "data1", "keras")
    if not res["backend"] == "keras":
        raise AssertionError
    if not res["model"] == "model1":
        raise AssertionError
    if not res["dataset"] == "data1":
        raise AssertionError
    if "metrics" not in res:
        raise AssertionError


def test_evaluate_maxtext(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test evaluate with maxtext."""
    monkeypatch.setattr("gemma_4_sql.backends.maxtext.evaluate.generate_sql", lambda *_args, **_kwargs: {"sql": "SELECT 1"})
    res = evaluate("model1", "data1", "maxtext")
    if not res["backend"] == "maxtext":
        raise AssertionError
    if not res["model"] == "model1":
        raise AssertionError
    if not res["dataset"] == "data1":
        raise AssertionError
    if "metrics" not in res:
        raise AssertionError


def test_evaluate_pytorch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test evaluate with pytorch."""
    monkeypatch.setattr("gemma_4_sql.backends.pytorch.evaluate.generate_sql", lambda *_args, **_kwargs: {"sql": "SELECT 1"})
    res = evaluate("model1", "data1", "pytorch")
    if not res["backend"] == "pytorch":
        raise AssertionError
    if not res["model"] == "model1":
        raise AssertionError
    if not res["dataset"] == "data1":
        raise AssertionError
    if "metrics" not in res:
        raise AssertionError


@pytest.mark.usefixtures("monkeypatch")
def test_evaluate_invalid() -> None:
    """Test evaluate with invalid backend."""
    with pytest.raises(ValueError, match="Unknown backend: invalid"):
        evaluate("model1", "data1", "invalid")

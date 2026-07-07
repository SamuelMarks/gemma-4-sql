"""Tests for SDK Evaluation module."""

import pytest

from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.sdk.evaluation import evaluate


def test_evaluate_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test evaluate with jax."""
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    jax_agent = get_backend("jax")
    monkeypatch.setattr(jax_agent, "generate_sql", lambda *_args, **_kwargs: {"sql": "SELECT 1"})

    def raise_err(*a, **k):
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("Mocked missing JAX")

    monkeypatch.setattr(jax_agent, "build_dataloader", raise_err)
    with pytest.raises(DependencyMissingError):
        evaluate("model1", "data1", "jax")


def test_evaluate_keras(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test evaluate with keras.

    Raises:
        AssertionError: Description.

    """
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    keras_agent = get_backend("keras")
    monkeypatch.setattr(keras_agent, "build_dataloader", lambda *args, **kwargs: {"loader": [{"inputs": [[1]], "targets": [[1]]}]})
    monkeypatch.setattr(keras_agent, "generate_sql", lambda *_args, **_kwargs: {"sql": "SELECT 1"})
    res = evaluate("model1", "data1", "keras")
    if res["backend"] != "keras":
        raise AssertionError
    if res["model"] != "model1":
        raise AssertionError
    if res["dataset"] != "data1":
        raise AssertionError
    if "metrics" not in res:
        raise AssertionError


def test_evaluate_maxtext(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test evaluate with maxtext.

    Raises:
        AssertionError: Description.

    """
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    maxtext_agent = get_backend("maxtext")
    monkeypatch.setattr(maxtext_agent, "generate_sql", lambda *_args, **_kwargs: {"sql": "SELECT 1"})
    monkeypatch.setattr(maxtext_agent, "build_dataloader", lambda *_args, **_kwargs: {"loader": [{"inputs": [[1]], "targets": [[2]]}]})
    res = evaluate("model1", "data1", "maxtext")
    if res["backend"] != "maxtext":
        raise AssertionError
    if res["model"] != "model1":
        raise AssertionError
    if res["dataset"] != "data1":
        raise AssertionError
    if "metrics" not in res:
        raise AssertionError


def test_evaluate_pytorch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test evaluate with pytorch.

    Raises:
        AssertionError: Description.

    """
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    pytorch_agent = get_backend("pytorch")
    monkeypatch.setattr(pytorch_agent, "generate_sql", lambda *_args, **_kwargs: {"sql": "SELECT 1"})
    monkeypatch.setattr(pytorch_agent, "build_dataloader", lambda *_args, **_kwargs: {"loader": [([[1]], [[2]])]})
    res = evaluate("model1", "data1", "pytorch")
    if res["backend"] != "pytorch":
        raise AssertionError
    if res["model"] != "model1":
        raise AssertionError
    if res["dataset"] != "data1":
        raise AssertionError
    if "metrics" not in res:
        raise AssertionError


@pytest.mark.usefixtures("monkeypatch")
def test_evaluate_invalid() -> None:
    """Test evaluate with invalid backend."""
    with pytest.raises(ValueError, match="Unknown backend: invalid"):
        evaluate("model1", "data1", "invalid")

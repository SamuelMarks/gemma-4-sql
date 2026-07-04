# Copyright 2024
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

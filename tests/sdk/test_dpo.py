# Copyright 2024
"""Tests for SDK DPO module."""

import pytest

from gemma_4_sql.sdk.dpo import run_dpo


def test_run_dpo_pytorch() -> None:
    """Initialize function test_run_dpo_pytorch.

    Raises:
        AssertionError: Description.

    """
    res = run_dpo("model1", "data1", "pytorch", 0.1)
    if not res["backend"] == "pytorch":
        raise AssertionError
    if not res["action"] == "dpo":
        raise AssertionError
    if not res["model"] == "model1":
        raise AssertionError


def test_run_dpo_jax() -> None:
    """Initialize function test_run_dpo_jax.

    Raises:
        AssertionError: Description.

    """
    res = run_dpo("model2", "data2", "jax", 0.2)
    if not res["backend"] == "jax":
        raise AssertionError
    if not res["action"] == "dpo":
        raise AssertionError
    if not res["model"] == "model2":
        raise AssertionError


def test_run_dpo_keras() -> None:
    """Initialize function test_run_dpo_keras.

    Raises:
        AssertionError: Description.

    """
    res = run_dpo("model3", "data3", "keras", 0.3)
    if not res["backend"] == "keras":
        raise AssertionError
    if not res["action"] == "dpo":
        raise AssertionError
    if not res["model"] == "model3":
        raise AssertionError


def test_run_dpo_maxtext() -> None:
    """Initialize function test_run_dpo_maxtext.

    Raises:
        AssertionError: Description.

    """
    res = run_dpo("model4", "data4", "maxtext", 0.4)
    if not res["backend"] == "maxtext":
        raise AssertionError
    if not res["action"] == "dpo":
        raise AssertionError
    if not res["model"] == "model4":
        raise AssertionError


def test_run_dpo_unknown_backend() -> None:
    """Initialize function test_run_dpo_unknown_backend."""
    with pytest.raises(ValueError, match="Unknown backend: missing"):
        run_dpo("my-model", "my-data", "missing")

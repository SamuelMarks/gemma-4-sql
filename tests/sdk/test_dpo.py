"""Tests for DPO SDK module."""

import pytest

from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.sdk.dpo import run_dpo


def test_run_dpo_jax() -> None:
    """Initialize function test_run_dpo_jax.

    Raises:
        AssertionError: Description.

    """
    res = run_dpo("model1", "data1", "jax", 0.1)
    if not res["backend"] == "jax":
        raise AssertionError
    if not res["action"] == "dpo":
        raise AssertionError
    if not res["model"] == "model1":
        raise AssertionError
    if not res["dataset"] == "data1":
        raise AssertionError
    expected_beta = 0.1
    if not res["beta"] == expected_beta:
        raise AssertionError


def test_run_dpo_pytorch() -> None:
    """Initialize function test_run_dpo_pytorch.

    Raises:
        AssertionError: Description.

    """
    res = run_dpo(model_name="model2", dataset="data2", beta=0.2, backend="pytorch")
    if not res["backend"] == "pytorch":
        raise AssertionError
    if not res["action"] == "dpo":
        raise AssertionError
    if not res["model"] == "model2":
        raise AssertionError
    if not res["dataset"] == "data2":
        raise AssertionError
    expected_beta = 0.2
    if not res["beta"] == expected_beta:
        raise AssertionError


def test_run_dpo_keras() -> None:
    """Initialize function test_run_dpo_keras.

    Raises:
        AssertionError: Description.

    """
    res = run_dpo(model_name="model3", dataset="data3", backend="keras", beta=0.3)
    if not res["backend"] == "keras":
        raise AssertionError
    if not res["action"] == "dpo":
        raise AssertionError
    if not res["model"] == "model3":
        raise AssertionError
    expected_beta = 0.3
    if not res["beta"] == expected_beta:
        raise AssertionError


def test_run_dpo_maxtext() -> None:
    """Initialize function test_run_dpo_maxtext."""
    with pytest.raises(DependencyMissingError):
        run_dpo("model4", "data4", "maxtext", 0.4)


def test_run_dpo_invalid() -> None:
    """Initialize function test_run_dpo_invalid."""
    with pytest.raises(ValueError, match=r".*"):
        run_dpo("model", "data", "unknown")

"""Tests for SDK DPO module."""

import pytest
from gemma_4_sql.sdk.dpo import run_dpo


def test_run_dpo_pytorch() -> None:
    """Initialize function test_run_dpo_pytorch."""
    res = run_dpo("model1", "data1", backend="pytorch", beta=0.1)
    if not res["backend"] == "pytorch":
        raise AssertionError
    if not res["action"] == "dpo":
        raise AssertionError
    if not res["model"] == "model1":
        raise AssertionError


def test_run_dpo_jax() -> None:
    """Initialize function test_run_dpo_jax."""
    res = run_dpo("model2", "data2", backend="jax", beta=0.2)
    if not res["backend"] == "jax":
        raise AssertionError
    if not res["action"] == "dpo":
        raise AssertionError
    if not res["model"] == "model2":
        raise AssertionError


def test_run_dpo_keras() -> None:
    """Initialize function test_run_dpo_keras."""
    res = run_dpo("model3", "data3", backend="keras", beta=0.3)
    if not res["backend"] == "keras":
        raise AssertionError
    if not res["action"] == "dpo":
        raise AssertionError
    if not res["model"] == "model3":
        raise AssertionError


def test_run_dpo_maxtext() -> None:
    """Initialize function test_run_dpo_maxtext."""
    res = run_dpo("model4", "data4", backend="maxtext", beta=0.4)
    if not res["backend"] == "maxtext":
        raise AssertionError
    if not res["action"] == "dpo":
        raise AssertionError
    if not res["model"] == "model4":
        raise AssertionError


def test_run_dpo_unknown_backend() -> None:
    """Initialize function test_run_dpo_unknown_backend."""
    with pytest.raises(ValueError, match="Unknown backend: missing"):
        run_dpo("my-model", "my-data", backend="missing")

"""
Tests for SDK DPO module.
"""

import pytest

from gemma_4_sql.sdk.dpo import run_dpo


def test_run_dpo_pytorch() -> None:
    res = run_dpo("model1", "data1", backend="pytorch", beta=0.1)
    assert res["backend"] == "pytorch"
    assert res["action"] == "dpo"
    assert res["model"] == "model1"


def test_run_dpo_jax() -> None:
    res = run_dpo("model2", "data2", backend="jax", beta=0.2)
    assert res["backend"] == "jax"
    assert res["action"] == "dpo"
    assert res["model"] == "model2"


def test_run_dpo_keras() -> None:
    res = run_dpo("model3", "data3", backend="keras", beta=0.3)
    assert res["backend"] == "keras"
    assert res["action"] == "dpo"
    assert res["model"] == "model3"


def test_run_dpo_maxtext() -> None:
    res = run_dpo("model4", "data4", backend="maxtext", beta=0.4)
    assert res["backend"] == "maxtext"
    assert res["action"] == "dpo"
    assert res["model"] == "model4"


def test_run_dpo_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unknown backend: missing"):
        run_dpo("my-model", "my-data", backend="missing")

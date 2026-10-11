"""Tests for DPO SDK module."""

import pytest

from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.sdk.dpo import run_dpo


def test_run_dpo_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    jax_agent = get_backend("jax")
    monkeypatch.setattr(jax_agent, "run_dpo", lambda cfg, **k: {"backend": "jax", "action": "dpo", "model": cfg.model_name, "dataset": cfg.dataset, "beta": cfg.beta})

    """Initialize function test_run_dpo_jax.

    Raises:
        AssertionError: Description.

    """
    res = run_dpo("model1", "data1", "jax", 0.1)
    assert res["backend"] == "jax"
    assert res["action"] == "dpo"
    assert res["model"] == "model1"
    assert res["dataset"] == "data1"
    expected_beta = 0.1
    assert res["beta"] == expected_beta


def test_run_dpo_pytorch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Initialize function test_run_dpo_pytorch."""
    from unittest.mock import MagicMock

    import gemma_4_sql.sdk.dpo as sdk_dpo
    from gemma_4_sql.sdk import registry

    mock_be = MagicMock()
    mock_be.run_dpo.return_value = {"backend": "pytorch", "action": "dpo", "model": "model2", "dataset": "data2", "beta": 0.2}
    monkeypatch.setattr(registry, "get_backend", lambda x: mock_be)

    res = sdk_dpo.run_dpo(model_name="model2", dataset="data2", beta=0.2, backend="pytorch")

    assert res["backend"] == "pytorch"
    assert res["action"] == "dpo"
    assert res["model"] == "model2"
    assert res["dataset"] == "data2"
    expected_beta = 0.2
    assert res["beta"] == expected_beta


def test_run_dpo_keras(monkeypatch: pytest.MonkeyPatch) -> None:
    """Initialize function test_run_dpo_keras."""
    import gemma_4_sql.backends.keras as kb
    import gemma_4_sql.backends.keras.dpo as kdpo

    monkeypatch.setattr(kdpo, "tf", None)
    with pytest.raises(DependencyMissingError):
        run_dpo(model_name="model3", dataset="data3", backend="keras", beta=0.3)

    def mock_dpo(*args, **kwargs):
        """Docstring for mock_dpo."""
        return {
            "backend": "keras",
            "action": "dpo",
            "model": "model3",
            "dataset": "data3",
            "beta": 0.3,
        }

    monkeypatch.setattr(kdpo, "keras", object())
    monkeypatch.setattr(kdpo, "tf", object())
    monkeypatch.setattr(kdpo, "run_dpo", mock_dpo)
    monkeypatch.setattr(kb, "run_dpo", mock_dpo)

    res = run_dpo(model_name="model3", dataset="data3", backend="keras", beta=0.3)
    assert res["backend"] == "keras"
    assert res["action"] == "dpo"
    assert res["model"] == "model3"
    expected_beta = 0.3
    assert res["beta"] == expected_beta


def test_run_dpo_maxtext() -> None:
    """Initialize function test_run_dpo_maxtext."""
    with pytest.raises(DependencyMissingError):
        run_dpo("model4", "data4", "maxtext", 0.4)


def test_run_dpo_invalid() -> None:
    """Initialize function test_run_dpo_invalid."""
    with pytest.raises(ValueError, match=r".*"):
        run_dpo("model", "data", "unknown")

"""Behavioral unit tests for the DPO SDK module."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from gemma_4_sql.exceptions import ConfigurationError, DependencyMissingError
from gemma_4_sql.sdk.dpo import run_dpo
from gemma_4_sql.type_hints import DPOConfig


def test_run_dpo_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `run_dpo` constructs a `DPOConfig` and forwards `**kwargs` to the JAX backend."""
    from gemma_4_sql.sdk.registry import get_backend

    jax_agent = get_backend("jax")
    captured_cfg: list[DPOConfig] = []
    captured_kw: dict[str, Any] = {}

    def _spy_dpo(cfg: DPOConfig, **kw: Any) -> dict[str, Any]:
        """Record DPOConfig and keyword arguments."""
        captured_cfg.append(cfg)
        captured_kw.update(kw)
        return {
            "backend": "jax",
            "action": "dpo",
            "model": cfg.model_name,
            "dataset": cfg.dataset,
            "beta": cfg.beta,
            "status": "completed",
            "final_loss": 0.25,
        }

    monkeypatch.setattr(jax_agent, "run_dpo", _spy_dpo)

    res = run_dpo("model1", "data1", "jax", 0.1, epochs=3, learning_rate=2e-5, batch_size=4, custom_flag="enabled")
    assert res["backend"] == "jax"
    assert res["action"] == "dpo"
    assert res["model"] == "model1"
    assert res["dataset"] == "data1"
    assert res["beta"] == 0.1
    assert len(captured_cfg) == 1
    assert captured_cfg[0].epochs == 3
    assert captured_cfg[0].learning_rate == 2e-5
    assert captured_cfg[0].batch_size == 4
    assert captured_cfg[0].extra_kwargs == {"custom_flag": "enabled"}
    assert captured_kw == {}


def test_run_dpo_pytorch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `run_dpo` passes an accurately populated `DPOConfig` to the PyTorch backend."""
    import gemma_4_sql.sdk.dpo as sdk_dpo

    mock_be = MagicMock()
    mock_be.run_dpo.return_value = {
        "backend": "pytorch",
        "action": "dpo",
        "model": "model2",
        "dataset": "data2",
        "beta": 0.2,
        "status": "completed",
        "final_loss": 0.1,
    }
    monkeypatch.setattr(sdk_dpo, "get_backend", lambda _x: mock_be)

    res = sdk_dpo.run_dpo(
        model_name="model2",
        dataset="data2",
        beta=0.2,
        epochs=2,
        learning_rate=5e-5,
        batch_size=8,
        backend="pytorch",
    )

    assert res["backend"] == "pytorch"
    assert res["action"] == "dpo"
    assert res["model"] == "model2"
    assert res["dataset"] == "data2"
    assert res["beta"] == 0.2

    mock_be.run_dpo.assert_called_once()
    passed_cfg = mock_be.run_dpo.call_args.args[0]
    assert isinstance(passed_cfg, DPOConfig)
    assert passed_cfg.model_name == "model2"
    assert passed_cfg.dataset == "data2"
    assert passed_cfg.beta == 0.2
    assert passed_cfg.epochs == 2
    assert passed_cfg.learning_rate == 5e-5
    assert passed_cfg.batch_size == 8


def test_run_dpo_keras(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `run_dpo` with the Keras backend propagates `DependencyMissingError` and returns DPO status."""
    import gemma_4_sql.backends.keras as kb
    import gemma_4_sql.backends.keras.dpo as kdpo

    monkeypatch.setattr(kdpo, "tf", None)
    with pytest.raises(DependencyMissingError):
        run_dpo(model_name="model3", dataset="data3", backend="keras", beta=0.3)

    def mock_dpo(cfg: DPOConfig, **_kwargs: Any) -> dict[str, Any]:
        """Return simulated Keras DPO dictionary."""
        return {
            "backend": "keras",
            "action": "dpo",
            "model": cfg.model_name,
            "dataset": cfg.dataset,
            "beta": cfg.beta,
            "status": "completed",
            "final_loss": 0.15,
        }

    monkeypatch.setattr(kdpo, "keras", object())
    monkeypatch.setattr(kdpo, "tf", object())
    monkeypatch.setattr(kdpo, "run_dpo", mock_dpo)
    monkeypatch.setattr(kb, "run_dpo", mock_dpo)

    res = run_dpo(model_name="model3", dataset="data3", backend="keras", beta=0.3)
    assert res["backend"] == "keras"
    assert res["action"] == "dpo"
    assert res["model"] == "model3"
    assert res["beta"] == 0.3


def test_run_dpo_maxtext() -> None:
    """Verify `run_dpo` with MaxText raises `DependencyMissingError` when MaxText is uninstalled."""
    with pytest.raises(DependencyMissingError):
        run_dpo("model4", "data4", "maxtext", 0.4)


def test_run_dpo_invalid() -> None:
    """Verify `run_dpo` raises `ConfigurationError` (a `ValueError` subclass) on an unknown backend."""
    with pytest.raises(ConfigurationError, match="Unknown backend: unknown"):
        run_dpo("model", "data", "unknown")

"""Tests for SDK ETL module."""

import pytest

from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.sdk.etl import _route_backend, etl_posttrain, etl_pretrain, etl_sft
from gemma_4_sql.type_hints import ETLConfig


def test_route_backend_invalid() -> None:
    """Test routing with invalid backend."""
    with pytest.raises(ValueError, match="Unknown backend: invalid"):
        _route_backend(ETLConfig(dataset_name="dummy/data", split="train", batch_size=16, distributed=False), "invalid")


def test_route_backend_pytorch() -> None:
    """Test routing with pytorch backend.

    Raises:
        AssertionError: Description.

    """
    res = _route_backend(ETLConfig(dataset_name="dummy/data", split="train", batch_size=16, distributed=False), "pytorch")
    if not res["backend"] == "pytorch":
        raise AssertionError


def test_etl_pretrain(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test etl_pretrain helper."""
    config = ETLConfig(dataset_name="dummy/data", split="train", batch_size=16, distributed=False)
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    jax_agent = get_backend("jax")

    def raise_err(*a, **k):
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("Mocked missing JAX")

    monkeypatch.setattr(jax_agent, "build_dataloader", raise_err)
    with pytest.raises(DependencyMissingError):
        etl_pretrain(config, backend="jax")


def test_etl_sft(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test etl_sft helper.

    Raises:
        AssertionError: Description.

    """
    config = ETLConfig(dataset_name="dummy/data", split="train", batch_size=16, distributed=False)
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    keras_agent = get_backend("keras")
    monkeypatch.setattr(keras_agent, "build_dataloader", lambda *a, **k: {"backend": "keras", "dataset": "dummy/data", "batch_size": 16})
    res = etl_sft(config, backend="keras")
    if not res["dataset"] == "dummy/data":
        raise AssertionError
    if not res["batch_size"] == int("16"):
        raise AssertionError


def test_etl_posttrain(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test etl_posttrain helper.

    Raises:
        AssertionError: Description.

    """
    config = ETLConfig(dataset_name="dummy/data", split="train", batch_size=16, distributed=False)
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    maxtext_agent = get_backend("maxtext")
    monkeypatch.setattr(maxtext_agent, "build_dataloader", lambda *a, **k: {"backend": "maxtext", "dataset": "dummy/data", "batch_size": 16})
    res = etl_posttrain(config, backend="maxtext")
    if not res["dataset"] == "dummy/data":
        raise AssertionError
    if not res["batch_size"] == int("16"):
        raise AssertionError


import pytest

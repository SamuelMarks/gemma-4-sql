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


def test_etl_pretrain() -> None:
    """Test etl_pretrain helper."""
    config = ETLConfig(dataset_name="dummy/data", split="train", batch_size=16, distributed=False)
    with pytest.raises(DependencyMissingError):
        etl_pretrain(config, backend="jax")


def test_etl_sft() -> None:
    """Test etl_sft helper.

    Raises:
        AssertionError: Description.

    """
    config = ETLConfig(dataset_name="dummy/data", split="train", batch_size=16, distributed=False)
    res = etl_sft(config, backend="keras")
    if not res["dataset"] == "dummy/data":
        raise AssertionError
    if not res["batch_size"] == int("16"):
        raise AssertionError


def test_etl_posttrain() -> None:
    """Test etl_posttrain helper.

    Raises:
        AssertionError: Description.

    """
    config = ETLConfig(dataset_name="dummy/data", split="train", batch_size=16, distributed=False)
    res = etl_posttrain(config, backend="maxtext")
    if not res["dataset"] == "dummy/data":
        raise AssertionError
    if not res["batch_size"] == int("16"):
        raise AssertionError

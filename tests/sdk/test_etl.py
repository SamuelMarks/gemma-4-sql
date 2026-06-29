"""Tests for SDK ETL module."""

import pytest
from gemma_4_sql.sdk.etl import _route_backend, etl_posttrain, etl_pretrain, etl_sft


def test_route_backend_invalid() -> None:
    """Test routing with invalid backend."""
    with pytest.raises(ValueError, match="Unknown backend: invalid"):
        _route_backend("dummy/data", "train", 16, "invalid", distributed=False)


def test_route_backend_pytorch() -> None:
    """Test routing with pytorch backend."""
    res = _route_backend("dummy/data", "train", 16, "pytorch", distributed=False)
    if not res["backend"] == "pytorch":
        raise AssertionError


def test_etl_pretrain() -> None:
    """Test etl_pretrain helper."""
    res = etl_pretrain("dummy/data", "train", 16, "jax", distributed=False)
    if not res["dataset"] == "dummy/data":
        raise AssertionError
    if not res["batch_size"] == int("16"):
        raise AssertionError


def test_etl_sft() -> None:
    """Test etl_sft helper."""
    res = etl_sft("dummy/data", "train", 16, "keras", distributed=False)
    if not res["dataset"] == "dummy/data":
        raise AssertionError
    if not res["batch_size"] == int("16"):
        raise AssertionError


def test_etl_posttrain() -> None:
    """Test etl_posttrain helper."""
    res = etl_posttrain("dummy/data", "train", 16, "maxtext", distributed=False)
    if not res["dataset"] == "dummy/data":
        raise AssertionError
    if not res["batch_size"] == int("16"):
        raise AssertionError

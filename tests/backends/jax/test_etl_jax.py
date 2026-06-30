"""Module docstring."""

import sys
from unittest import mock

import pytest


@pytest.fixture(autouse=True)
def _clean_sys_modules() -> object:
    """Initialize function clean_sys_modules."""
    keys = list(sys.modules.keys())
    yield
    for k in list(sys.modules.keys()):
        if k not in keys and "gemma_4_sql" in k:
            del sys.modules[k]


"Tests for JAX ETL module."


def test_jax_etl_mocked() -> None:
    """Test JAX ETL when libraries are missing via direct assignment."""
    etl_jax = __import__("gemma_4_sql.backends.jax.etl", fromlist=[""])
    original_datasets = getattr(etl_jax, "datasets", None)
    original_grain = getattr(etl_jax, "grain", None)
    try:
        etl_jax.datasets = None  # type: ignore[attr-defined]
        etl_jax.grain = None  # type: ignore[attr-defined]
        res = etl_jax.build_dataloader("test", "train", 10)
        if not res["status"] == "mocked":
            raise AssertionError
        if not res["backend"] == "jax":
            raise AssertionError
    finally:
        etl_jax.datasets = original_datasets  # type: ignore[attr-defined]
        etl_jax.grain = original_grain  # type: ignore[attr-defined]


def test_jax_etl_import_error() -> None:
    """Test JAX ETL ImportError fallback."""
    with mock.patch.dict(sys.modules, {"datasets": None, "grain": None, "grain.python": None}):
        etl_jax = __import__("gemma_4_sql.backends.jax.etl", fromlist=[""])
        res = etl_jax.build_dataloader("test", "train", 10)
        if not res["status"] == "mocked":
            raise AssertionError


class MockDatasets:
    """Initialize class MockDatasets."""

    @staticmethod
    def load_dataset(*_args: object, **_kwargs: object) -> list[dict]:  # type: ignore[type-arg]
        """Initialize function load_dataset.

        Args:
        ----
        name: Description of name.
        split: Description of split.

        """
        return [{"question": "Q1", "query": "A1"}, {"sql_prompt": "Q2", "sql": "A2"}]


class MockGrain:
    """Initialize class MockGrain."""

    class RandomAccessDataSource:
        """Initialize class RandomAccessDataSource."""

    class MapTransform:
        """Initialize class MapTransform."""

    @staticmethod
    def no_sharding() -> str:
        """Initialize function nosharding."""
        return "no_sharding"

    @staticmethod
    def jax_distributed_sharding() -> str:
        """Initialize function jaxdistributedsharding."""
        return "jax_distributed_sharding"

    @staticmethod
    def index_sampler(*_args: object, **kwargs: object) -> str:
        """Initialize function indexsampler.

        Args:
        ----
        args: Description of args.
        kwargs: Description of kwargs.

        """
        return kwargs.get("shard_options", "sampler")  # type: ignore[return-value]

    @staticmethod
    def batch_func(batch_size: int) -> str:
        """Initialize function batch.

        Args:
        ----
        batch_size: Description of batch_size.

        """
        return f"batch_{batch_size}"

    class DataLoader:
        """Initialize class DataLoader."""

        def __init__(self: object, data_source: object, sampler: object, operations: object) -> None:
            """Initialize function __init__.

            Args:
            ----
            data_source: Description of data_source.
            sampler: Description of sampler.
            operations: Description of operations.

            """
            self.data_source = data_source
            self.sampler = sampler
            self.operations = operations


def _check_res(res: dict, status: str, *, distributed: bool, sampler: str) -> None:  # type: ignore[type-arg]
    if res["status"] != status:
        raise AssertionError
    if distributed is not None and res.get("distributed") is not distributed:
        raise AssertionError
    if sampler is not None and getattr(res.get("loader"), "sampler", None) != sampler:
        raise AssertionError


def test_jax_etl_loaded() -> None:
    """Test JAX ETL when libraries are present."""
    etl = __import__("gemma_4_sql.backends.jax.etl", fromlist=[""])
    original_datasets = getattr(etl, "datasets", None)
    original_grain = getattr(etl, "grain", None)
    try:
        etl.datasets = MockDatasets()  # type: ignore[attr-defined]
        etl.grain = MockGrain()  # type: ignore[attr-defined]
        res = etl.build_dataloader("test", "train", 10, distributed=False)
        _check_res(res, "loaded", distributed=False, sampler="no_sharding")
    finally:
        etl.datasets = original_datasets  # type: ignore[attr-defined]
        etl.grain = original_grain  # type: ignore[attr-defined]


def test_jax_etl_dist() -> None:
    """Test JAX ETL distributed."""
    etl = __import__("gemma_4_sql.backends.jax.etl", fromlist=[""])
    original_datasets = getattr(etl, "datasets", None)
    original_grain = getattr(etl, "grain", None)
    try:
        etl.datasets = MockDatasets()  # type: ignore[attr-defined]
        etl.grain = MockGrain()  # type: ignore[attr-defined]
        res_dist_ = etl.build_dataloader("test", "train", 10, distributed=True)
        _check_res(res_dist_, "loaded", distributed=True, sampler="jax_distributed_sharding")
    finally:
        etl.datasets = original_datasets  # type: ignore[attr-defined]
        etl.grain = original_grain  # type: ignore[attr-defined]

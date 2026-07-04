# Copyright 2024
"""Provide module docstring."""

import sys
from unittest import mock

import pytest

from gemma_4_sql.type_hints import ETLConfig


@pytest.fixture(autouse=True)
def _clean_sys_modules() -> object:
    """Initialize function clean_sys_modules.

    Yields:
        object: Description of yield.

    """
    sys = __import__("sys", fromlist=[""])
    keys = list(sys.modules.keys())
    yield
    for k in list(sys.modules.keys()):
        if k not in keys and "gemma_4_sql" in k:
            del sys.modules[k]


"Tests for Keras ETL module."


def test_keras_etl_mocked() -> None:
    """Test Keras ETL when libraries are missing via direct assignment.

    Raises:
        AssertionError: Description.

    """
    etl_keras = __import__("gemma_4_sql.backends.keras.etl", fromlist=[""])
    original_datasets = getattr(etl_keras, "datasets", None)
    original_grain = getattr(etl_keras, "grain", None)
    try:
        etl_keras.datasets = None
        etl_keras.grain = None
        res = etl_keras.build_dataloader(ETLConfig(dataset_name="test", split="train", batch_size=10))
        if not res["status"] == "mocked":
            raise AssertionError
        if not res["backend"] == "keras":
            raise AssertionError
    finally:
        etl_keras.datasets = original_datasets
        etl_keras.grain = original_grain


def test_keras_etl_import_error() -> None:
    """Test Keras ETL ImportError fallback.

    Raises:
        AssertionError: Description.

    """
    if "gemma_4_sql.backends.keras.etl" in sys.modules:
        del sys.modules["gemma_4_sql.backends.keras.etl"]
    with mock.patch.dict(sys.modules, {"datasets": None, "grain": None, "grain.python": None}):
        etl_keras = __import__("gemma_4_sql.backends.keras.etl", fromlist=[""])
        res = etl_keras.build_dataloader(ETLConfig(dataset_name="test", split="train", batch_size=10))
        if not res["status"] == "mocked":
            raise AssertionError


class MockDatasets:
    """Initialize class MockDatasets."""

    @staticmethod
    def load_dataset(*_args: object, **_kwargs: object) -> list[dict]:
        """Initialize function load_dataset.

        Args:
        ----
        name: Description of name.
        split: Description of split.


        Returns:
            object: Description of return.

        """
        return [{"question": "Q1", "query": "A1"}]


class MockGrain:
    """Initialize class MockGrain."""

    class RandomAccessDataSource:
        """Initialize class RandomAccessDataSource."""

    class MapTransform:
        """Initialize class MapTransform."""

    @staticmethod
    def no_sharding() -> str:
        """Initialize function nosharding.

        Returns:
            object: Description of return.

        """
        return "no_sharding"

    @staticmethod
    def jax_distributed_sharding() -> str:
        """Initialize function jaxdistributedsharding.

        Returns:
            object: Description of return.

        """
        return "jax_distributed_sharding"

    @staticmethod
    def index_sampler(*_args: object, **kwargs: object) -> str:
        """Initialize function indexsampler.

        Args:
        ----
        args: Description of args.
        kwargs: Description of kwargs.


        Returns:
            object: Description of return.

        """
        return kwargs.get("shard_options", "sampler")

    @staticmethod
    def batch(batch_size: int) -> str:
        """Initialize function batch.

        Args:
        ----
        batch_size: Description of batch_size.


        Returns:
            object: Description of return.

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


MockGrain.NoSharding = staticmethod(MockGrain.no_sharding)
MockGrain.JAXDistributedSharding = staticmethod(MockGrain.jax_distributed_sharding)
MockGrain.IndexSampler = staticmethod(MockGrain.index_sampler)
MockGrain.Batch = staticmethod(MockGrain.batch)


def test_keras_etl_loaded() -> None:
    """Test Keras ETL when libraries are present.

    Raises:
        AssertionError: Description.

    """
    etl_keras = __import__("gemma_4_sql.backends.keras.etl", fromlist=[""])
    original_datasets = getattr(etl_keras, "datasets", None)
    original_grain = getattr(etl_keras, "grain", None)
    try:
        etl_keras.datasets = MockDatasets()
        etl_keras.grain = MockGrain()
        res = etl_keras.build_dataloader(ETLConfig(dataset_name="test", split="train", batch_size=10, distributed=False))
        if not res["status"] == "loaded":
            raise AssertionError
        if res["distributed"] is not False:
            raise AssertionError
        if not res["loader"].sampler == "no_sharding":
            raise AssertionError
        res_dist = etl_keras.build_dataloader(ETLConfig(dataset_name="test", split="train", batch_size=10, distributed=True))
        if not res_dist["status"] == "loaded":
            raise AssertionError
        if res_dist["distributed"] is not True:
            raise AssertionError
        if not res_dist["loader"].sampler == "jax_distributed_sharding":
            raise AssertionError
        loader = res["loader"]
        if not len(loader.data_source) == 1:
            raise AssertionError
        if not loader.data_source[0] == {"question": "Q1", "query": "A1"}:
            raise AssertionError
        loader.operations[0]
    finally:
        etl_keras.datasets = original_datasets
        etl_keras.grain = original_grain


def test_etl_keras_imports_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    importlib = __import__("importlib", fromlist=[""])
    sys = __import__("sys", fromlist=[""])
    mdl = __import__("gemma_4_sql.backends.keras.etl", fromlist=[""])
    monkeypatch.setitem(sys.modules, "keras", None)
    importlib.reload(mdl)
    monkeypatch.undo()
    monkeypatch.setitem(sys.modules, "duckdb", None)
    importlib.reload(mdl)
    monkeypatch.undo()
    monkeypatch.setitem(sys.modules, "tensorflow", None)
    importlib.reload(mdl)
    monkeypatch.undo()
    importlib.reload(mdl)

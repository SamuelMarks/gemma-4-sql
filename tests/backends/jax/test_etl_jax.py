"""Module docstring."""

import sys
from unittest import mock

import pytest

import gemma_4_sql.backends.jax.etl as jax_etl


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
        etl_jax.datasets = None
        etl_jax.grain = None
        res = etl_jax.build_dataloader("test", "train", 10)
        if not res["status"] == "mocked":
            raise TypeError
        if not res["backend"] == "jax":
            raise TypeError
    finally:
        etl_jax.datasets = original_datasets
        etl_jax.grain = original_grain


def test_jax_etl_import_error() -> None:
    """Test JAX ETL ImportError fallback."""
    with mock.patch.dict(sys.modules, {"datasets": None, "grain": None, "grain.python": None}):
        if "gemma_4_sql.backends.jax.etl" in sys.modules:
            del sys.modules["gemma_4_sql.backends.jax.etl"]
        etl_jax = __import__("gemma_4_sql.backends.jax.etl", fromlist=[""])
        res = etl_jax.build_dataloader("test", "train", 10)
        if not res["status"] == "mocked":
            raise TypeError


class MockDatasets:
    """Initialize class MockDatasets."""

    @staticmethod
    def load_dataset(*_args: object, **_kwargs: object) -> list[dict]:
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
    def mock_no_sharding() -> str:
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
        return kwargs.get("shard_options", "sampler")

    @staticmethod
    def _b(batch_size: int) -> str:
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


MockGrain.NoSharding = staticmethod(MockGrain.no_sharding)
MockGrain.JAXDistributedSharding = staticmethod(MockGrain.jax_distributed_sharding)
MockGrain.IndexSampler = staticmethod(MockGrain.index_sampler)
MockGrain.Batch = staticmethod(MockGrain.batch)
MockGrain.NoSharding = staticmethod(MockGrain.no_sharding)
MockGrain.JAXDistributedSharding = staticmethod(MockGrain.jax_distributed_sharding)
MockGrain.IndexSampler = staticmethod(MockGrain.index_sampler)
MockGrain.Batch = staticmethod(MockGrain.batch)


def _check_res(res: dict, status: str, *, distributed: bool, sampler: str) -> None:
    """Execute function."""
    if res["status"] != status:
        raise TypeError
    if distributed is not None and res.get("distributed") is not distributed:
        raise TypeError
    if sampler is not None and getattr(res.get("loader"), "sampler", None) != sampler:
        raise TypeError


def test_jax_etl_loaded() -> None:
    """Test JAX ETL when libraries are present."""
    etl = __import__("gemma_4_sql.backends.jax.etl", fromlist=[""])
    original_datasets = getattr(etl, "datasets", None)
    original_grain = getattr(etl, "grain", None)
    try:
        etl.datasets = MockDatasets()
        etl.grain = MockGrain()
        res = etl.build_dataloader("test", "train", 10, distributed=False)
        _check_res(res, "loaded", distributed=False, sampler="no_sharding")
    finally:
        etl.datasets = original_datasets
        etl.grain = original_grain


def test_jax_etl_dist() -> None:
    """Test JAX ETL distributed."""
    etl = __import__("gemma_4_sql.backends.jax.etl", fromlist=[""])
    original_datasets = getattr(etl, "datasets", None)
    original_grain = getattr(etl, "grain", None)
    try:
        etl.datasets = MockDatasets()
        etl.grain = MockGrain()
        res_dist_ = etl.build_dataloader("test", "train", 10, distributed=True)
        _check_res(res_dist_, "loaded", distributed=True, sampler="jax_distributed_sharding")
    finally:
        etl.datasets = original_datasets
        etl.grain = original_grain


def _dummy() -> object:
    """Execute function."""


def test_jax_etl_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    monkeypatch.setattr(jax_etl, "datasets", "mock")
    monkeypatch.setattr(jax_etl, "grain", "mock")

    class MockDatasets2:
        """Provide class docstring."""

        def load_dataset(self, _name: object, _split: object) -> object:
            """Execute function."""
            return [{"question": "q", "query": "a"}]

    class MockGrain:
        """Provide class docstring."""

        class RandomAccessDataSource:
            """Provide class docstring."""

        class MapTransform:
            """Provide class docstring."""

        class IndexSampler:
            """Provide class docstring."""

            def __init__(self, **kwargs: object) -> object:
                """Execute function."""

        class DataLoader:
            """Provide class docstring."""

            def __init__(self, **kwargs: object) -> object:
                """Execute function."""
                self.data_source = kwargs.get("data_source")

        def _b(self, **_kwargs: object) -> object:
            """Execute function."""
            return "batch"

        def mock_no_sharding(self) -> object:
            """Execute function."""
            return "no_sharding"

        def mock_jax_sharding(self) -> object:
            """Execute function."""
            return "jax_sharding"

    monkeypatch.setattr(jax_etl, "datasets", MockDatasets2())
    monkeypatch.setattr(jax_etl, "grain", MockGrain())
    res = jax_etl.build_dataloader("ds", "split", batch_size=2)
    if not (res["status"] == "loaded"):
        raise TypeError


def test_jax_etl_duckdb_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    monkeypatch.setattr(jax_etl, "datasets", "mock")
    monkeypatch.setattr(jax_etl, "grain", "mock")
    monkeypatch.setattr(jax_etl, "duckdb", None)
    with pytest.raises(ImportError, match="duckdb is required"):
        jax_etl.build_dataloader("ds", "split", duckdb_path="path", duckdb_table="table")

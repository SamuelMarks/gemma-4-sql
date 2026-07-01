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
        etl_jax.datasets = None
        etl_jax.grain = None
        res = etl_jax.build_dataloader("test", "train", 10)
        if not res["status"] == "mocked":
            raise AssertionError
        if not res["backend"] == "jax":
            raise AssertionError
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
    def index_sampler(
        *_args: object,
        **kwargs: object,
    ) -> str:
        """Initialize function indexsampler.

        Args:
        ----
        args: Description of args.
        kwargs: Description of kwargs.

        """
        return kwargs.get("shard_options", "sampler")  # type: ignore[return-value]

    @staticmethod
    def batch(
        batch_size: int,
    ) -> str:
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


def _dummy():
    pass


import gemma_4_sql.backends.jax.etl as jax_etl  # noqa: E402


def test_jax_etl_real(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jax_etl, "datasets", "mock")
    monkeypatch.setattr(jax_etl, "grain", "mock")

    class MockDatasets:
        def load_dataset(self, name, split):
            return [{"question": "q", "query": "a"}]

    class MockGrain:
        class RandomAccessDataSource:
            pass

        class MapTransform:
            pass

        class IndexSampler:
            def __init__(self, **kwargs):
                pass

        class DataLoader:
            def __init__(self, **kwargs):
                self.data_source = kwargs.get("data_source")

        def Batch(self, **kwargs):
            return "batch"

        def NoSharding(self):
            return "no_sharding"

        def JAXDistributedSharding(self):
            return "jax_sharding"

    monkeypatch.setattr(jax_etl, "datasets", MockDatasets())
    monkeypatch.setattr(jax_etl, "grain", MockGrain())

    res = jax_etl.build_dataloader("ds", "split", batch_size=2)
    assert res["status"] == "loaded"


def test_jax_etl_duckdb_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jax_etl, "datasets", "mock")
    monkeypatch.setattr(jax_etl, "grain", "mock")
    monkeypatch.setattr(jax_etl, "duckdb", None)

    with pytest.raises(ImportError, match="duckdb is required"):
        jax_etl.build_dataloader("ds", "split", duckdb_path="path", duckdb_table="table")


def test_jax_etl_duckdb_real(monkeypatch: pytest.MonkeyPatch) -> None:
    class MockDatasets:
        def load_dataset(self, name, split):
            return [{"question": "q", "query": "a"}]

    class MockGrain:
        class RandomAccessDataSource:
            pass

        class MapTransform:
            pass

        class IndexSampler:
            def __init__(self, **kwargs):
                pass

        class DataLoader:
            def __init__(self, **kwargs):
                self.data_source = kwargs.get("data_source")

        def Batch(self, **kwargs):
            return "batch"

        def NoSharding(self):
            return "no_sharding"

        def JAXDistributedSharding(self):
            return "jax_sharding"

    class MockDuckDB:
        def connect(self, path, read_only):
            class MockConn:
                def execute(self, query, params):
                    class MockCursor:
                        def fetchdf(self):
                            class MockDF:
                                def to_dict(self, orient):
                                    return [{"question": "db_q", "query": "db_a"}]

                            return MockDF()

                    return MockCursor()

                def close(self):
                    pass

            return MockConn()

    monkeypatch.setattr(jax_etl, "datasets", MockDatasets())
    monkeypatch.setattr(jax_etl, "grain", MockGrain())
    monkeypatch.setattr(jax_etl, "duckdb", MockDuckDB())

    res = jax_etl.build_dataloader("ds", "split", duckdb_path="path", duckdb_table="table", distributed=True)
    assert res["status"] == "loaded"


def test_jax_etl_transforms(monkeypatch: pytest.MonkeyPatch) -> None:
    class MockDatasets:
        def load_dataset(self, name, split):
            return [{"question": "q", "query": "a"}, {"sql_prompt": "sp", "sql": "s"}]

    class MockGrain:
        class RandomAccessDataSource:
            pass

        class MapTransform:
            pass

        class IndexSampler:
            def __init__(self, **kwargs):
                pass

        class DataLoader:
            def __init__(self, **kwargs):
                self.ops = kwargs.get("operations")
                self.ds = kwargs.get("data_source")

        def Batch(self, **kwargs):
            return "batch"

        def NoSharding(self):
            return "no_sharding"

    monkeypatch.setattr(jax_etl, "datasets", MockDatasets())
    monkeypatch.setattr(jax_etl, "grain", MockGrain())

    res = jax_etl.build_dataloader("ds", "split")
    loader = res["loader"]

    ds = loader.ds
    assert len(ds) == 2
    assert ds[0]["question"] == "q"

    transform = loader.ops[0]
    out1 = transform.map(ds[0])
    out2 = transform.map(ds[1])
    assert isinstance(out1["inputs"], list)


def test_etl_imports_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib
    import sys

    # Mock duckdb import failure
    monkeypatch.setitem(sys.modules, "duckdb", None)
    importlib.reload(jax_etl)

    # Restore original to not break other tests
    monkeypatch.undo()
    importlib.reload(jax_etl)

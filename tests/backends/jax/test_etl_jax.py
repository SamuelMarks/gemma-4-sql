# Copyright 2024
"""Module docstring."""

import sys
from unittest import mock

import pytest

import gemma_4_sql.backends.jax.etl as jax_etl
from gemma_4_sql.type_hints import ETLConfig


@pytest.fixture(autouse=True)
def _clean_sys_modules() -> object:
    """Initialize function clean_sys_modules.

    Yields:
        object: Description of yield.

    """
    keys = list(sys.modules.keys())
    yield
    for k in list(sys.modules.keys()):
        if k not in keys and "gemma_4_sql" in k:
            del sys.modules[k]


"Tests for JAX ETL module."


def test_jax_etl_mocked() -> None:
    """Test JAX ETL when libraries are missing via direct assignment.

    Raises:
        TypeError: Description.

    """
    etl_jax = __import__("gemma_4_sql.backends.jax.etl", fromlist=[""])
    original_datasets = getattr(etl_jax, "datasets", None)
    original_grain = getattr(etl_jax, "grain", None)
    try:
        etl_jax.datasets = None
        etl_jax.grain = None
        res = etl_jax.build_dataloader(ETLConfig(dataset_name="test", split="train", batch_size=10))
        if not res["status"] == "mocked":
            raise TypeError
        if not res["backend"] == "jax":
            raise TypeError
    finally:
        etl_jax.datasets = original_datasets
        etl_jax.grain = original_grain


def test_jax_etl_import_error() -> None:
    """Test JAX ETL ImportError fallback.

    Raises:
        TypeError: Description.

    """
    with mock.patch.dict(sys.modules, {"datasets": None, "grain": None, "grain.python": None}):
        if "gemma_4_sql.backends.jax.etl" in sys.modules:
            del sys.modules["gemma_4_sql.backends.jax.etl"]
        etl_jax = __import__("gemma_4_sql.backends.jax.etl", fromlist=[""])
        res = etl_jax.build_dataloader(ETLConfig(dataset_name="test", split="train", batch_size=10))
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


        Returns:
            object: Description of return.

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
    def _b(batch_size: int) -> str:
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


MockGrain.NoSharding = staticmethod(MockGrain.mock_no_sharding)
MockGrain.JAXDistributedSharding = staticmethod(MockGrain.jax_distributed_sharding)
MockGrain.IndexSampler = staticmethod(MockGrain.index_sampler)
MockGrain.Batch = staticmethod(MockGrain._b)
MockGrain.NoSharding = staticmethod(MockGrain.mock_no_sharding)
MockGrain.JAXDistributedSharding = staticmethod(MockGrain.jax_distributed_sharding)
MockGrain.IndexSampler = staticmethod(MockGrain.index_sampler)
MockGrain.Batch = staticmethod(MockGrain._b)


def _check_res(res: dict, status: str, *, distributed: bool, sampler: str) -> None:
    """Execute function.

    Raises:
        TypeError: Description.

    """
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
        res = etl.build_dataloader(ETLConfig(dataset_name="test", split="train", batch_size=10, distributed=False))
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
        res_dist_ = etl.build_dataloader(ETLConfig(dataset_name="test", split="train", batch_size=10, distributed=True))
        _check_res(res_dist_, "loaded", distributed=True, sampler="jax_distributed_sharding")
    finally:
        etl.datasets = original_datasets
        etl.grain = original_grain


def _dummy() -> object:
    """Execute function."""


def test_jax_etl_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        TypeError: Description.

    """
    monkeypatch.setattr(jax_etl, "datasets", "mock")
    monkeypatch.setattr(jax_etl, "grain", type("MockGrain", (), {"IndexSampler": lambda *args, **kwargs: None, "DataLoader": lambda *args, **kwargs: None, "Batch": lambda *args, **kwargs: None, "RandomAccessDataSource": object, "MapTransform": object}))

    class MockDatasets2:
        """Provide class docstring."""

        def load_dataset(self, *_args: object, **_kwargs: object) -> object:
            """Execute function.

            Returns:
                object: Description of return.

            """
            return [{"question": "q", "query": "a"}]

    class MockGrain:
        """Provide class docstring."""

        class Batch:
            def __init__(self, **kwargs: object) -> None:
                pass

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
            """Execute function.

            Returns:
                object: Description of return.

            """
            return "batch"

        def mock_no_sharding(self) -> object:
            """Execute function.

            Returns:
                object: Description of return.

            """
            return "no_sharding"

        def mock_jax_sharding(self) -> object:
            """Execute function.

            Returns:
                object: Description of return.

            """
            return "jax_sharding"

    monkeypatch.setattr(jax_etl, "datasets", MockDatasets2())
    monkeypatch.setattr(jax_etl, "grain", MockGrain())
    res = jax_etl.build_dataloader(ETLConfig(dataset_name="ds", split="split", batch_size=2))
    if not (res["status"] == "loaded"):
        raise TypeError


def test_jax_etl_duckdb_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    monkeypatch.setattr(jax_etl, "datasets", "mock")
    monkeypatch.setattr(jax_etl, "grain", type("MockGrain", (), {"IndexSampler": lambda *args, **kwargs: None, "DataLoader": lambda *args, **kwargs: None, "Batch": lambda *args, **kwargs: None, "RandomAccessDataSource": object, "MapTransform": object}))
    import sys

    sys.modules["gemma_4_sql.backends.common_data"].duckdb = None
    monkeypatch.setattr(jax_etl, "_load_duckdb_dataset", lambda *args, **kwargs: (_ for _ in ()).throw(ImportError("duckdb is required")))
    with pytest.raises(ImportError, match="duckdb is required"):
        jax_etl.build_dataloader(ETLConfig(dataset_name="ds", split="split", duckdb_path="path", duckdb_table="table"))


def test_jax_etl_duckdb_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test DuckDB dataset loading with jax_etl."""

    class MockDuckDB:
        def connect(self, *args: object, **kwargs: object) -> object:
            class MockConn:
                def execute(self, *args: object, **kwargs: object) -> object:
                    class MockResult:
                        def fetchdf(self) -> object:
                            class MockDF:
                                def to_dict(self, orient: str = "records") -> object:
                                    return [{"question": "q1", "query": "a1"}, {"sql_prompt": "q2", "sql": "a2"}]

                            return MockDF()

                    return MockResult()

                def close(self) -> None:
                    pass

            return MockConn()

    monkeypatch.setattr(jax_etl, "duckdb", MockDuckDB())

    class MockGrain:
        class RandomAccessDataSource:
            pass

        class MapTransform:
            pass

        class IndexSampler:
            def __init__(self, **kwargs: object) -> None:
                pass

        class DataLoader:
            def __init__(self, **kwargs: object) -> None:
                pass

        def NoSharding(self) -> str:
            return "no_sharding"

        class Batch:
            def __init__(self, **kwargs: object) -> None:
                pass

    monkeypatch.setattr(jax_etl, "grain", MockGrain())
    monkeypatch.setattr(jax_etl, "datasets", "mock")

    res = jax_etl.build_dataloader(ETLConfig(dataset_name="ds", split="split", duckdb_path="path", duckdb_table="table"))
    assert res["status"] == "loaded"


def test_jax_etl_grain_classes() -> None:
    """Test Grain inner classes."""

    class MockGrain:
        class RandomAccessDataSource:
            pass

        class MapTransform:
            pass

    HFDataSource, JAXFormatTransform = jax_etl._get_grain_classes(MockGrain)

    ds = HFDataSource([{"a": 1}, {"a": 2}])
    assert len(ds) == 2
    assert ds[0] == {"a": 1}
    assert ds[1] == {"a": 2}

    class MockTokenizer:
        def encode(self, x: str) -> list[int]:
            return [len(x)]

    transform = JAXFormatTransform(MockTokenizer())
    res1 = transform.map({"question": "hello", "query": "world"})
    assert res1 == {"inputs": [5], "targets": [5]}

    res2 = transform.map({"sql_prompt": "hi", "sql": "bye"})
    assert res2 == {"inputs": [2], "targets": [3]}

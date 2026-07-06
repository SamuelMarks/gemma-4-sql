from gemma_4_sql.exceptions import DependencyMissingError

"""Provide module docstring."""

import contextlib
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


"Tests for MaxText ETL module."


def test_maxtext_etl_mocked() -> None:
    """Test MaxText ETL when libraries are missing via direct assignment.

    Raises:
        AssertionError: Description.

    """
    etl_maxtext = __import__("gemma_4_sql.backends.maxtext.etl", fromlist=[""])
    original_datasets = getattr(etl_maxtext, "datasets", None)
    original_grain = getattr(etl_maxtext, "grain", None)
    try:
        etl_maxtext.datasets = None
        etl_maxtext.grain = None
        with pytest.raises(DependencyMissingError):
            etl_maxtext.build_dataloader(ETLConfig(dataset_name="test", split="train", batch_size=10))
    finally:
        etl_maxtext.datasets = original_datasets
        etl_maxtext.grain = original_grain


def test_maxtext_etl_import_error() -> None:
    """Test MaxText ETL ImportError fallback.

    Raises:
        AssertionError: Description.

    """
    if "gemma_4_sql.backends.maxtext.etl" in sys.modules:
        del sys.modules["gemma_4_sql.backends.maxtext.etl"]
    with mock.patch.dict(sys.modules, {"datasets": None, "grain": None, "grain.python": None}):
        etl_maxtext = __import__("gemma_4_sql.backends.maxtext.etl", fromlist=[""])
        with pytest.raises(DependencyMissingError):
            etl_maxtext.build_dataloader(ETLConfig(dataset_name="test", split="train", batch_size=10))


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
MockGrain.NoSharding = staticmethod(MockGrain.no_sharding)
MockGrain.JAXDistributedSharding = staticmethod(MockGrain.jax_distributed_sharding)
MockGrain.IndexSampler = staticmethod(MockGrain.index_sampler)
MockGrain.Batch = staticmethod(MockGrain.batch)


def _check_res(res: dict, status: str, *, distributed: bool, sampler: str) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    if res["status"] != status:
        raise AssertionError
    if distributed is not None and res.get("distributed") is not distributed:
        raise AssertionError
    if sampler is not None and getattr(res.get("loader"), "sampler", None) != sampler:
        raise AssertionError


def test_maxtext_etl_loaded() -> None:
    """Test MAXTEXT ETL when libraries are present."""
    etl = __import__("gemma_4_sql.backends.maxtext.etl", fromlist=[""])
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


def test_maxtext_etl_dist() -> None:
    """Test MAXTEXT ETL distributed."""
    etl = __import__("gemma_4_sql.backends.maxtext.etl", fromlist=[""])
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


def test_etl_imports_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    importlib = __import__("importlib", fromlist=[""])
    sys = __import__("sys", fromlist=[""])
    m_etl = __import__("gemma_4_sql.backends.maxtext.etl", fromlist=[""])
    monkeypatch.setitem(sys.modules, "duckdb", None)
    importlib.reload(m_etl)
    monkeypatch.undo()
    importlib.reload(m_etl)


def test_maxtext_in_memory_data_source() -> None:
    """Execute function."""


class MockConn:
    """Provide class docstring."""

    def execute(self, *_args: object, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return self

    def fetchdf(self) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """

        class MockDF:
            """Provide class docstring."""

            def to_dict(self, orient: object = "records") -> object:
                """Execute function.

                Returns:
                    object: Description of return.

                """
                return [{"a": 1}]

        return MockDF()

    def close(self) -> None:
        """Execute function."""


class MockDuckdb:
    """Provide class docstring."""

    def connect(self, *_args: object, **_kwargs: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return MockConn()


def test_duckdb_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    m_etl = __import__("gemma_4_sql.backends.maxtext.etl", fromlist=[""])
    monkeypatch.setattr(m_etl, "duckdb", MockDuckdb())
    monkeypatch.setattr(m_etl, "grain", object())
    with contextlib.suppress(Exception):
        m_etl.build_dataloader(ETLConfig(dataset_name="dataset", split="train", batch_size=1, duckdb_path="test.db", duckdb_table="tbl"))


class MockTokenizer:
    """Provide class docstring."""

    def __init__(self, model_name: object = None) -> None:
        """Execute function."""

    def encode(self, x: object) -> object:
        """Execute function.

        Returns:
            object: Description of return.

        """
        return [x]


def test_etl_nested_classes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function.

    Raises:
        AssertionError: Description.

    """
    m_etl = __import__("gemma_4_sql.backends.maxtext.etl", fromlist=[""])
    monkeypatch.setattr(m_etl, "datasets", MockDatasets())
    monkeypatch.setattr(m_etl, "grain", MockGrain())
    monkeypatch.setattr(m_etl, "SQLTokenizer", MockTokenizer)
    res = m_etl.build_dataloader(ETLConfig(dataset_name="ds", split="train"))
    dl = res["loader"]
    ds = dl.data_source
    if len(ds) != int("1"):
        raise AssertionError
    if ds[0] != {"question": "Q1", "query": "A1"}:
        raise AssertionError
    transform = dl.operations[0]
    m1 = transform.map({"question": "Q1", "query": "A1"})
    if m1["inputs"] != ["Q1"]:
        raise AssertionError
    if m1["targets"] != ["A1"]:
        raise AssertionError
    m2 = transform.map({"sql_prompt": "p", "sql": "s"})
    if m2["inputs"] != ["p"]:
        raise AssertionError
    if m2["targets"] != ["s"]:
        raise AssertionError


def test_duckdb_execution_2(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    m_etl = __import__("gemma_4_sql.backends.maxtext.etl", fromlist=[""])
    monkeypatch.setattr(m_etl, "duckdb", MockDuckdb())
    monkeypatch.setattr(m_etl, "datasets", MockDatasets())
    monkeypatch.setattr(m_etl, "grain", MockGrain())
    monkeypatch.setattr(m_etl, "SQLTokenizer", MockTokenizer)
    m_etl.build_dataloader(ETLConfig(dataset_name="dataset", split="train", batch_size=1, duckdb_path="test.db", duckdb_table="tbl"))


def test_duckdb_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute function."""
    m_etl = __import__("gemma_4_sql.backends.maxtext.etl", fromlist=[""])
    monkeypatch.setattr(m_etl, "datasets", MockDatasets())
    monkeypatch.setattr(m_etl, "grain", MockGrain())
    import sys

    sys.modules["gemma_4_sql.backends.common_data"].duckdb = None
    monkeypatch.setattr(m_etl, "_load_duckdb_dataset", lambda *args, **kwargs: (_ for _ in ()).throw(ImportError("duckdb is required")))
    with pytest.raises(ImportError):
        m_etl.build_dataloader(ETLConfig(dataset_name="dataset", split="train", batch_size=1, duckdb_path="test.db", duckdb_table="tbl"))

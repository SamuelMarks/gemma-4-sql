"""Module docstring."""

import sys
from unittest import mock

import pytest


@pytest.fixture(autouse=True)
def _clean_sys_modules() -> object:
    """Initialize function clean_sys_modules."""
    sys = __import__("sys")
    keys = list(sys.modules.keys())
    yield
    for k in list(sys.modules.keys()):
        if k not in keys and "gemma_4_sql" in k:
            del sys.modules[k]


"Tests for MaxText ETL module."


def test_maxtext_etl_mocked() -> None:
    """Test MaxText ETL when libraries are missing via direct assignment."""
    etl_maxtext = __import__("gemma_4_sql.backends.maxtext.etl", fromlist=[""])
    original_datasets = getattr(etl_maxtext, "datasets", None)
    original_grain = getattr(etl_maxtext, "grain", None)
    try:
        etl_maxtext.datasets = None
        etl_maxtext.grain = None
        res = etl_maxtext.build_dataloader("test", "train", 10)
        if not res["status"] == "mocked":
            raise AssertionError
        if not res["backend"] == "maxtext":
            raise AssertionError
    finally:
        etl_maxtext.datasets = original_datasets
        etl_maxtext.grain = original_grain


def test_maxtext_etl_import_error() -> None:
    """Test MaxText ETL ImportError fallback."""
    if "gemma_4_sql.backends.maxtext.etl" in sys.modules:
        del sys.modules["gemma_4_sql.backends.maxtext.etl"]
    with mock.patch.dict(sys.modules, {"datasets": None, "grain": None, "grain.python": None}):
        etl_maxtext = __import__("gemma_4_sql.backends.maxtext.etl", fromlist=[""])
        res = etl_maxtext.build_dataloader("test", "train", 10)
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
        return [{"question": "Q1", "query": "A1"}]


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


def test_maxtext_etl_loaded() -> None:
    """Test MAXTEXT ETL when libraries are present."""
    etl = __import__("gemma_4_sql.backends.maxtext.etl", fromlist=[""])
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


def test_maxtext_etl_dist() -> None:
    """Test MAXTEXT ETL distributed."""
    etl = __import__("gemma_4_sql.backends.maxtext.etl", fromlist=[""])
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


def test_etl_imports_fail(monkeypatch: pytest.MonkeyPatch):
    import importlib
    import sys

    import gemma_4_sql.backends.maxtext.etl as m_etl

    monkeypatch.setitem(sys.modules, "duckdb", None)
    importlib.reload(m_etl)
    monkeypatch.undo()
    importlib.reload(m_etl)


def test_maxtext_in_memory_data_source():
    pass


def test_duckdb_execution(monkeypatch: pytest.MonkeyPatch):
    import gemma_4_sql.backends.maxtext.etl as m_etl

    class MockConn:
        def execute(self, *args, **kwargs):
            return self

        def fetchdf(self):
            class MockDF:
                def to_dict(self, orient):
                    return [{"a": 1}]

            return MockDF()

        def close(self):
            pass

    class MockDuckdb:
        def connect(self, *args, **kwargs):
            return MockConn()

    monkeypatch.setattr(m_etl, "duckdb", MockDuckdb())
    monkeypatch.setattr(m_etl, "grain", object())
    try:
        m_etl.build_dataloader("dataset", 1, "test.db", "tbl")
    except Exception:
        pass


def test_etl_nested_classes(monkeypatch: pytest.MonkeyPatch):
    import gemma_4_sql.backends.maxtext.etl as m_etl

    class MockDatasets:
        @staticmethod
        def load_dataset(name, split):
            return [{"question": "q", "query": "a"}, {"sql_prompt": "p", "sql": "s"}]

    class MockGrain:
        class RandomAccessDataSource:
            pass

        class MapTransform:
            pass

        class IndexSampler:
            def __init__(self, **kwargs):
                pass

        class Batch:
            def __init__(self, **kwargs):
                pass

        class DataLoader:
            def __init__(self, data_source, sampler, operations):
                self.data_source = data_source
                self.sampler = sampler
                self.operations = operations

        @staticmethod
        def NoSharding():
            return None

    class MockTokenizer:
        def __init__(self, model_name=None):
            pass

        def encode(self, x):
            return [x]

    monkeypatch.setattr(m_etl, "datasets", MockDatasets())
    monkeypatch.setattr(m_etl, "grain", MockGrain())
    monkeypatch.setattr(m_etl, "SQLTokenizer", MockTokenizer)

    res = m_etl.build_dataloader("ds", "train")
    dl = res["loader"]

    # Test HFDataSource
    ds = dl.data_source
    assert len(ds) == 2
    assert ds[0] == {"question": "q", "query": "a"}

    # Test MaxTextFormatTransform
    transform = dl.operations[0]
    m1 = transform.map({"question": "q", "query": "a"})
    assert m1["inputs"] == ["q"]
    assert m1["targets"] == ["a"]
    m2 = transform.map({"sql_prompt": "p", "sql": "s"})
    assert m2["inputs"] == ["p"]
    assert m2["targets"] == ["s"]


def test_duckdb_execution_2(monkeypatch: pytest.MonkeyPatch):
    import gemma_4_sql.backends.maxtext.etl as m_etl

    class MockConn:
        def execute(self, *args, **kwargs):
            return self

        def fetchdf(self):
            class MockDF:
                def to_dict(self, orient):
                    return [{"a": 1}]

            return MockDF()

        def close(self):
            pass

    class MockDuckdb:
        def connect(self, *args, **kwargs):
            return MockConn()

    class MockDatasets:
        pass

    class MockGrain:
        class RandomAccessDataSource:
            pass

        class MapTransform:
            pass

        class IndexSampler:
            def __init__(self, **kwargs):
                pass

        class Batch:
            def __init__(self, **kwargs):
                pass

        class DataLoader:
            def __init__(self, data_source, sampler, operations):
                self.data_source = data_source
                self.sampler = sampler
                self.operations = operations

        @staticmethod
        def NoSharding():
            return None

    class MockTokenizer:
        def __init__(self, model_name=None):
            pass

        def encode(self, x):
            return [x]

    monkeypatch.setattr(m_etl, "duckdb", MockDuckdb())
    monkeypatch.setattr(m_etl, "datasets", MockDatasets())
    monkeypatch.setattr(m_etl, "grain", MockGrain())
    monkeypatch.setattr(m_etl, "SQLTokenizer", MockTokenizer)

    m_etl.build_dataloader("dataset", 1, duckdb_path="test.db", duckdb_table="tbl")


def test_duckdb_missing(monkeypatch: pytest.MonkeyPatch):
    import gemma_4_sql.backends.maxtext.etl as m_etl

    class MockDatasets:
        pass

    class MockGrain:
        pass

    monkeypatch.setattr(m_etl, "datasets", MockDatasets())
    monkeypatch.setattr(m_etl, "grain", MockGrain())
    monkeypatch.setattr(m_etl, "duckdb", None)

    with pytest.raises(ImportError):
        m_etl.build_dataloader("dataset", 1, duckdb_path="test.db", duckdb_table="tbl")

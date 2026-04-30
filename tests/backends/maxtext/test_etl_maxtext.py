"""Tests for MaxText ETL module."""

import sys
from unittest import mock


def test_maxtext_etl_mocked() -> None:
    """Test MaxText ETL when libraries are missing via direct assignment."""
    import gemma_4_sql.backends.maxtext.etl as etl_maxtext

    original_datasets = getattr(etl_maxtext, "datasets", None)
    original_grain = getattr(etl_maxtext, "grain", None)

    try:
        etl_maxtext.datasets = None
        etl_maxtext.grain = None

        res = etl_maxtext.build_dataloader("test", "train", 10)
        assert res["status"] == "mocked"
        assert res["backend"] == "maxtext"
    finally:
        etl_maxtext.datasets = original_datasets
        etl_maxtext.grain = original_grain


def test_maxtext_etl_import_error() -> None:
    """Test MaxText ETL ImportError fallback."""
    if "gemma_4_sql.backends.maxtext.etl" in sys.modules:
        del sys.modules["gemma_4_sql.backends.maxtext.etl"]

    with mock.patch.dict(
        sys.modules, {"datasets": None, "grain": None, "grain.python": None}
    ):
        import gemma_4_sql.backends.maxtext.etl as etl_maxtext

        res = etl_maxtext.build_dataloader("test", "train", 10)
        assert res["status"] == "mocked"


def test_maxtext_etl_loaded() -> None:
    """Test MaxText ETL when libraries are present."""

    class MockDatasets:
        @staticmethod
        def load_dataset(name: str, split: str) -> list[dict]:
            return [{"question": "Q1", "query": "A1"}]

    class MockGrain:
        class RandomAccessDataSource:
            pass

        class MapTransform:
            pass

        @staticmethod
        def NoSharding() -> str:
            return "no_sharding"

        @staticmethod
        def JAXDistributedSharding() -> str:
            return "jax_distributed_sharding"

        @staticmethod
        def IndexSampler(*args, **kwargs) -> str:
            return kwargs.get("shard_options", "sampler")

        @staticmethod
        def Batch(batch_size: int) -> str:
            return f"batch_{batch_size}"

        class DataLoader:
            def __init__(self, data_source, sampler, operations):
                self.data_source = data_source
                self.sampler = sampler
                self.operations = operations

    import gemma_4_sql.backends.maxtext.etl as etl_maxtext

    original_datasets = getattr(etl_maxtext, "datasets", None)
    original_grain = getattr(etl_maxtext, "grain", None)

    try:
        etl_maxtext.datasets = MockDatasets()  # type: ignore[attr-defined]
        etl_maxtext.grain = MockGrain()  # type: ignore[attr-defined]

        # Test non-distributed
        res = etl_maxtext.build_dataloader("test", "train", 10, distributed=False)
        assert res["status"] == "loaded"
        assert res["distributed"] is False
        assert res["loader"].sampler == "no_sharding"

        # Test distributed
        res_dist = etl_maxtext.build_dataloader("test", "train", 10, distributed=True)
        assert res_dist["status"] == "loaded"
        assert res_dist["distributed"] is True
        assert res_dist["loader"].sampler == "jax_distributed_sharding"

        loader = res["loader"]
        assert len(loader.data_source) == 1
        assert loader.data_source[0] == {"question": "Q1", "query": "A1"}

        transform = loader.operations[0]
        mapped = transform.map({"question": "Q1", "query": "A1"})
        assert mapped["inputs"] == [ord("Q"), ord("1")]
        assert mapped["targets"] == [ord("A"), ord("1")]
        assert mapped["segment_ids"] == [1]
        assert mapped["positions"] == [0]
    finally:
        etl_maxtext.datasets = original_datasets
        etl_maxtext.grain = original_grain

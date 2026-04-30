"""Tests for JAX ETL module."""

import sys
from unittest import mock


def test_jax_etl_mocked() -> None:
    """Test JAX ETL when libraries are missing via direct assignment."""
    import gemma_4_sql.backends.jax.etl as etl_jax

    original_datasets = getattr(etl_jax, "datasets", None)
    original_grain = getattr(etl_jax, "grain", None)

    try:
        etl_jax.datasets = None
        etl_jax.grain = None

        res = etl_jax.build_dataloader("test", "train", 10)
        assert res["status"] == "mocked"
        assert res["backend"] == "jax"
    finally:
        etl_jax.datasets = original_datasets
        etl_jax.grain = original_grain


def test_jax_etl_import_error() -> None:
    """Test JAX ETL ImportError fallback."""
    if "gemma_4_sql.backends.jax.etl" in sys.modules:
        del sys.modules["gemma_4_sql.backends.jax.etl"]

    with mock.patch.dict(
        sys.modules, {"datasets": None, "grain": None, "grain.python": None}
    ):
        import gemma_4_sql.backends.jax.etl as etl_jax

        res = etl_jax.build_dataloader("test", "train", 10)
        assert res["status"] == "mocked"


def test_jax_etl_loaded() -> None:
    """Test JAX ETL when libraries are present."""

    class MockDatasets:
        @staticmethod
        def load_dataset(name: str, split: str) -> list[dict]:
            return [
                {"question": "Q1", "query": "A1"},
                {"sql_prompt": "Q2", "sql": "A2"},
            ]

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

    import gemma_4_sql.backends.jax.etl as etl_jax

    original_datasets = getattr(etl_jax, "datasets", None)
    original_grain = getattr(etl_jax, "grain", None)

    try:
        etl_jax.datasets = MockDatasets()  # type: ignore[attr-defined]
        etl_jax.grain = MockGrain()  # type: ignore[attr-defined]

        # Test non-distributed
        res = etl_jax.build_dataloader("test", "train", 10, distributed=False)
        assert res["status"] == "loaded"
        assert res["distributed"] is False
        assert res["loader"].sampler == "no_sharding"

        # Test distributed
        res_dist = etl_jax.build_dataloader("test", "train", 10, distributed=True)
        assert res_dist["status"] == "loaded"
        assert res_dist["distributed"] is True
        assert res_dist["loader"].sampler == "jax_distributed_sharding"

        loader = res["loader"]
        assert len(loader.data_source) == 2
        assert loader.data_source[0] == {"question": "Q1", "query": "A1"}
        assert loader.sampler == "no_sharding"

        transform = loader.operations[0]
        assert transform.map({"question": "Q1", "query": "A1"}) == {
            "inputs": [ord("Q"), ord("1")],
            "targets": [ord("A"), ord("1")],
        }
        assert transform.map({"sql_prompt": "Q2", "sql": "A2"}) == {
            "inputs": [ord("Q"), ord("2")],
            "targets": [ord("A"), ord("2")],
        }
        assert transform.map({"sql_prompt": "Q2", "sql": "A2"}) == {
            "inputs": [ord("Q"), ord("2")],
            "targets": [ord("A"), ord("2")],
        }
    finally:
        etl_jax.datasets = original_datasets
        etl_jax.grain = original_grain

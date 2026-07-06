"""JAX-specific Grain ETL pipeline."""

from __future__ import annotations

import typing

from gemma_4_sql.backends.common_data import _get_grain_classes, _load_duckdb_dataset
from gemma_4_sql.backends.lazy_loader import catch_optional_imports
from gemma_4_sql.tokenization import SQLTokenizer
from gemma_4_sql.type_hints import ETLConfig

if typing.TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict, JSONValue
datasets = None
with catch_optional_imports():
    import datasets
grain = None
with catch_optional_imports():
    import grain.python as grain
duckdb = None
with catch_optional_imports():
    pass


def _load_hf_or_duckdb(dataset_name: str, split: str, duckdb_path: str | None, duckdb_table: str | None) -> object:
    """Load a dataset from Hugging Face or DuckDB.

    Args:
        dataset_name: The name of the Hugging Face dataset.
        split: The dataset split to load.
        duckdb_path: Optional path to a DuckDB database.
        duckdb_table: Optional name of the DuckDB table.

    Returns:
        The loaded dataset.
    """
    if duckdb_path and duckdb_table:
        return _load_duckdb_dataset(duckdb_path, duckdb_table)
    return datasets.load_dataset(dataset_name, split=split)


def _get_sampler(source_len: int, distributed: bool) -> object:
    """Get the appropriate Grain sampler for data loading.

    Args:
        source_len: The length of the source data.
        distributed: Whether to use distributed sharding.

    Returns:
        A Grain IndexSampler.
    """
    shard_options = getattr(grain, "JAXDistributedSharding", lambda: None)() if distributed else getattr(grain, "NoSharding", lambda: None)()
    return grain.IndexSampler(num_records=source_len, shard_options=shard_options, shuffle=False, num_epochs=1)


def build_dataloader(config: ETLConfig, **kwargs: JSONValue) -> JSONDict:
    """Build a JAX-specific Grain dataloader.

        Args:
                    **kwargs: Overrides for ETL configuration (e.g., duckdb_path, duckdb_table).
    config: The configuration parameters.

        Returns:
            A dictionary containing the results.
    """
    dataset_name = config.dataset_name
    split = config.split
    batch_size = config.batch_size
    distributed = config.distributed
    tokenizer_name = config.tokenizer_name
    duckdb_path = str(config.duckdb_path or kwargs.get("duckdb_path") or "")
    duckdb_table = str(config.duckdb_table or kwargs.get("duckdb_table") or "")

    if datasets is None or grain is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError(f"Missing grain or datasets. Cannot load {dataset_name}.")

    hf_dataset = _load_hf_or_duckdb(dataset_name, split, duckdb_path, duckdb_table)
    (data_source_cls, transform_cls) = _get_grain_classes(grain)
    source = data_source_cls(hf_dataset)
    tokenizer = SQLTokenizer(model_name=tokenizer_name)
    sampler = _get_sampler(len(source), distributed)
    dataloader = grain.DataLoader(data_source=source, sampler=sampler, operations=[transform_cls(tokenizer=tokenizer), grain.Batch(batch_size=batch_size)])
    return {"dataset": dataset_name, "split": split, "status": "loaded", "batch_size": batch_size, "backend": "jax", "distributed": distributed, "loader": dataloader}

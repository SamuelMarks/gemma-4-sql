# Copyright 2024
"""MaxText-specific Grain ETL pipeline."""

from __future__ import annotations

import typing

from gemma_4_sql.backends.common_data import _get_grain_classes, _load_duckdb_dataset
from gemma_4_sql.backends.lazy_loader import catch_optional_imports
from gemma_4_sql.tokenization import SQLTokenizer

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


def build_dataloader(config: object, **kwargs: JSONValue) -> JSONDict:
    """Build a MaxText-specific Grain dataloader.

    Returns:
        object: The resulting output from the operation.

    """
    dataset_name = getattr(config, "dataset_name", "dummy")
    split = getattr(config, "split", "train")
    batch_size = getattr(config, "batch_size", 32)
    distributed = getattr(config, "distributed", False)
    tokenizer_name = getattr(config, "tokenizer_name", None)
    duckdb_path = kwargs.get("duckdb_path") if not hasattr(config, "duckdb_path") else config.duckdb_path
    duckdb_table = kwargs.get("duckdb_table") if not hasattr(config, "duckdb_table") else config.duckdb_table
    if datasets is None or grain is None:
        return {"dataset": dataset_name, "split": split, "status": "mocked", "batch_size": batch_size, "backend": "maxtext", "distributed": distributed, "mock_samples": [{"query": "SELECT * FROM users", "nl": "Get all users"}]}
    hf_dataset = _load_duckdb_dataset(duckdb_path, duckdb_table) if duckdb_path and duckdb_table else datasets.load_dataset(dataset_name, split=split)
    (data_source_cls, transform_cls) = _get_grain_classes(grain)
    source = data_source_cls(hf_dataset)
    tokenizer = SQLTokenizer(model_name=tokenizer_name)
    shard_options = getattr(grain, "JAXDistributedSharding", lambda: None)() if distributed else getattr(grain, "NoSharding", lambda: None)()
    sampler = grain.IndexSampler(num_records=len(source), shard_options=shard_options, shuffle=False, num_epochs=1)
    dataloader = grain.DataLoader(data_source=source, sampler=sampler, operations=[transform_cls(tokenizer=tokenizer), grain.Batch(batch_size=batch_size)])
    return {"dataset": dataset_name, "split": split, "status": "loaded", "batch_size": batch_size, "backend": "maxtext", "distributed": distributed, "loader": dataloader}

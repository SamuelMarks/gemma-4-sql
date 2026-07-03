"""Keras-specific Grain ETL pipeline."""

from __future__ import annotations

import typing
from typing import cast

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
    import duckdb


def _load_duckdb_dataset(duckdb_path: str, duckdb_table: str) -> list[dict[str, object]]:
    """Load a dataset from DuckDB.

    Args:
    ----
        duckdb_path: Path to the DuckDB file.
        duckdb_table: Name of the table.

    Returns:
    -------
        A list of dictionaries representing the dataset.

    """
    if duckdb is None:
        msg = "duckdb is required for DuckDB support."
        raise ImportError(msg)
    conn = duckdb.connect(duckdb_path, read_only=True)
    try:
        hf_dataset = conn.execute("SELECT * FROM ?", (duckdb_table,)).fetchdf().to_dict(orient="records")
    finally:
        conn.close()
    return list(hf_dataset)


def _get_grain_classes(grain_module: object) -> tuple[object, object]:
    """Dynamically construct Grain classes.

    Args:
    ----
        grain_module: The loaded grain module.

    Returns:
    -------
        A tuple of (HFDataSource, KerasTupleTransform) classes.

    """
    base_ds = getattr(grain_module, "RandomAccessDataSource", object)
    base_map = getattr(grain_module, "MapTransform", object)
    any_base_ds = cast("object", base_ds)
    any_base_map = cast("object", base_map)

    class HFDataSource(any_base_ds):
        """Data source wrapping a Hugging Face dataset."""

        def __init__(self, hf_ds: object) -> None:
            """Execute function."""
            self._ds = hf_ds

        def __len__(self) -> int:
            """Execute function."""
            return len(self._ds)

        def __getitem__(self, idx: int) -> object:
            """Execute function."""
            return self._ds[idx]

    class KerasTupleTransform(any_base_map):
        """Transforms data into Keras (x, y) tuples."""

        def __init__(self, tokenizer: SQLTokenizer) -> None:
            """Execute function."""
            self.tokenizer = tokenizer

        def map(self, element: JSONDict) -> object:
            """Execute function."""
            x = element.get("sql_prompt", element.get("question", ""))
            y = element.get("sql", element.get("query", ""))
            return (self.tokenizer.encode(str(x)), self.tokenizer.encode(str(y)))

    return (HFDataSource, KerasTupleTransform)


def build_dataloader(dataset_name: str, split: str, batch_size: int = 32, *, distributed: bool = False, tokenizer_name: str | None = None, **kwargs: JSONValue) -> JSONDict:
    """Build a Keras-specific Grain dataloader.

    Args:
    ----
        dataset_name: The name of the dataset to load.
        split: The dataset split (e.g., 'train', 'test').
        batch_size: Number of items per batch.
        distributed: Whether to use distributed sharding.
        tokenizer_name: Optional name for the tokenizer model.
        **kwargs: Additional keyword arguments.

    Returns:
    -------
        A dictionary containing the built dataloader and dataset metadata.

    """
    duckdb_path = str(kwargs.get("duckdb_path", ""))
    duckdb_table = str(kwargs.get("duckdb_table", ""))
    if datasets is None or grain is None:
        return {"dataset": dataset_name, "split": split, "status": "mocked", "batch_size": batch_size, "backend": "keras", "distributed": distributed, "mock_samples": [{"query": "SELECT * FROM users", "nl": "Get all users"}]}
    hf_dataset = _load_duckdb_dataset(duckdb_path, duckdb_table) if duckdb_path and duckdb_table else datasets.load_dataset(dataset_name, split=split)
    (data_source_cls, transform_cls) = _get_grain_classes(grain)
    source = data_source_cls(hf_dataset)
    tokenizer = SQLTokenizer(model_name=tokenizer_name)
    shard_options = getattr(grain, "JAXDistributedSharding", lambda: None)() if distributed else getattr(grain, "NoSharding", lambda: None)()
    sampler = grain.IndexSampler(num_records=len(source), shard_options=shard_options, shuffle=False, num_epochs=1)
    dataloader = grain.DataLoader(data_source=source, sampler=sampler, operations=[transform_cls(tokenizer=tokenizer), grain.Batch(batch_size=batch_size)])
    return {"dataset": dataset_name, "split": split, "status": "loaded", "batch_size": batch_size, "backend": "keras", "distributed": distributed, "loader": dataloader}

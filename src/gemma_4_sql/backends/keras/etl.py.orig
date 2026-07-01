"""Keras-specific Grain ETL pipeline."""

from __future__ import annotations

import typing

from gemma_4_sql.tokenization import SQLTokenizer

try:
    import datasets
except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
    datasets = None
try:
    import grain.python as grain
except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
    grain = None
try:
    import duckdb
except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
    duckdb = None


def build_dataloader(dataset_name: str, split: str, batch_size: int = 32, *, distributed: bool = False, tokenizer_name: str | None = None, **kwargs: object) -> dict[str, object]:
    """Build a Keras-specific Grain dataloader.

    Args:
    ----
        dataset_name: The name of the dataset to load.
        split: The dataset split (e.g., 'train', 'test').
        batch_size: Number of items per batch.
        distributed: Whether to use distributed sharding.
        tokenizer_name: Optional name for the tokenizer model.
        duckdb_path: Optional path to a DuckDB database.
        duckdb_table: Optional table to load from DuckDB.
        **kwargs: Additional keyword arguments.

    Returns:
    -------
        A dictionary containing the built dataloader and dataset metadata.

    """
    duckdb_path = kwargs.get("duckdb_path")
    duckdb_table = kwargs.get("duckdb_table")
    if datasets is None or grain is None:
        return {"dataset": dataset_name, "split": split, "status": "mocked", "batch_size": batch_size, "backend": "keras", "distributed": distributed, "mock_samples": [{"query": "SELECT * FROM users", "nl": "Get all users"}]}
    if duckdb_path and duckdb_table:
        if duckdb is None:
            msg = "duckdb is required for DuckDB support."
            raise ImportError(msg)
        conn = duckdb.connect(duckdb_path, read_only=True)
        try:
            hf_dataset = conn.execute("SELECT * FROM ?", (duckdb_table,)).fetchdf().to_dict(orient="records")
        finally:
            conn.close()
    else:
        hf_dataset = datasets.load_dataset(dataset_name, split=split)

    class HFDataSource(grain.RandomAccessDataSource):  # type: ignore[misc]
        """Data source wrapping a Hugging Face dataset."""

        def __init__(self: typing.Any, hf_ds: object) -> None:
            """Initialize with dataset.

            Args:
            ----
                hf_ds: The Hugging Face dataset or list of records.

            """
            self._ds = hf_ds

        def __len__(self: typing.Any) -> int:
            """Return dataset length.

            Returns
            -------
                The total number of items in the dataset.

            """
            return len(self._ds)

        def __getitem__(self: typing.Any, idx: int) -> object:
            """Get dataset item by index.

            Args:
            ----
                idx: The index of the item to retrieve.

            Returns:
            -------
                The dataset element at the specified index.

            """
            return self._ds[idx]

    class KerasTupleTransform(grain.MapTransform):  # type: ignore[misc]
        """Transforms data into Keras (x, y) tuples."""

        def __init__(self: typing.Any, tokenizer: SQLTokenizer) -> None:
            """Initialize the transform with a tokenizer.

            Args:
            ----
                tokenizer: The SQLTokenizer to use for formatting.

            """
            self.tokenizer = tokenizer

        def map(self: typing.Any, element: dict[str, object]) -> object:
            """Map an element to a Keras (x, y) tuple.

            Args:
            ----
                element: A dictionary representing a single dataset example.

            Returns:
            -------
                A tuple of tokenized inputs (x) and targets (y).

            """
            x = element.get("sql_prompt", element.get("question", ""))
            y = element.get("sql", element.get("query", ""))
            return (self.tokenizer.encode(x), self.tokenizer.encode(y))

    source = HFDataSource(hf_dataset)
    tokenizer = SQLTokenizer(model_name=tokenizer_name)
    shard_options = getattr(grain, "JAXDistributedSharding", lambda: None)() if distributed else getattr(grain, "NoSharding", lambda: None)()
    sampler = grain.IndexSampler(num_records=len(source), shard_options=shard_options, shuffle=False, num_epochs=1)
    dataloader = grain.DataLoader(data_source=source, sampler=sampler, operations=[KerasTupleTransform(tokenizer=tokenizer), grain.Batch(batch_size=batch_size)])
    return {"dataset": dataset_name, "split": split, "status": "loaded", "batch_size": batch_size, "backend": "keras", "distributed": distributed, "loader": dataloader}

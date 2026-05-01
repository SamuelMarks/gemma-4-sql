"""
JAX-specific Grain ETL pipeline.
"""

from __future__ import annotations

from typing import Any

from gemma_4_sql.tokenization import SQLTokenizer

try:
    import datasets
except Exception:
    datasets = None

try:
    import grain.python as grain
except Exception:
    grain = None

try:
    import duckdb
except Exception:
    duckdb = None


def build_dataloader(
    dataset_name: str,
    split: str,
    batch_size: int = 32,
    distributed: bool = False,
    tokenizer_name: str | None = None,
    duckdb_path: str | None = None,
    duckdb_table: str | None = None,
) -> dict[str, Any]:
    """Builds a JAX-specific Grain dataloader."""
    if datasets is None or grain is None:
        return {
            "dataset": dataset_name,
            "split": split,
            "status": "mocked",
            "batch_size": batch_size,
            "backend": "jax",
            "distributed": distributed,
            "mock_samples": [{"query": "SELECT * FROM users", "nl": "Get all users"}],
        }

    if duckdb_path and duckdb_table:
        if duckdb is None:
            raise ImportError("duckdb is required for DuckDB support.")
        conn = duckdb.connect(duckdb_path, read_only=True)
        try:
            hf_dataset = (
                conn.execute(f"SELECT * FROM {duckdb_table}")
                .fetchdf()
                .to_dict(orient="records")
            )
        finally:
            conn.close()
    else:
        hf_dataset = datasets.load_dataset(dataset_name, split=split)

    class HFDataSource(grain.RandomAccessDataSource):
        """Data source wrapping a Hugging Face dataset."""

        def __init__(self, hf_ds: Any):
            """Initialize with dataset."""
            self._ds = hf_ds

        def __len__(self) -> int:
            """Return dataset length."""
            return len(self._ds)

        def __getitem__(self, idx: int) -> Any:
            """Get dataset item by index."""
            return self._ds[idx]

    class JAXFormatTransform(grain.MapTransform):
        """Transforms data into numpy/JAX compatible formats."""

        def __init__(self, tokenizer: SQLTokenizer):
            """Initialize the transform with a tokenizer."""
            self.tokenizer = tokenizer

        def map(self, element: dict[str, Any]) -> dict[str, Any]:
            """Map an element."""
            prompt = element.get("sql_prompt", element.get("question", ""))
            target = element.get("sql", element.get("query", ""))
            return {
                "inputs": self.tokenizer.encode(prompt),
                "targets": self.tokenizer.encode(target),
            }

    source = HFDataSource(hf_dataset)
    tokenizer = SQLTokenizer(model_name=tokenizer_name)

    if distributed:
        shard_options = getattr(grain, "JAXDistributedSharding", lambda: None)()
    else:
        shard_options = getattr(grain, "NoSharding", lambda: None)()

    sampler = grain.IndexSampler(
        num_records=len(source),
        shard_options=shard_options,
        shuffle=False,
        num_epochs=1,
    )

    dataloader = grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=[
            JAXFormatTransform(tokenizer=tokenizer),
            grain.Batch(batch_size=batch_size),
        ],
    )

    return {
        "dataset": dataset_name,
        "split": split,
        "status": "loaded",
        "batch_size": batch_size,
        "backend": "jax",
        "distributed": distributed,
        "loader": dataloader,
    }

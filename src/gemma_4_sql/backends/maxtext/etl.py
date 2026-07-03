"""MaxText-specific Grain ETL pipeline."""

from __future__ import annotations

import typing

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


def _load_duckdb_dataset(duckdb_path: str, duckdb_table: str) -> list[dict[str, typing.Any]]:
    """Load a dataset from DuckDB."""
    if duckdb is None:
        msg = "duckdb is required for DuckDB support."
        raise ImportError(msg)
    conn = duckdb.connect(duckdb_path, read_only=True)
    try:
        hf_dataset = conn.execute("SELECT * FROM ?", (duckdb_table,)).fetchdf().to_dict(orient="records")
    finally:
        conn.close()
    return list(hf_dataset)


def _get_grain_classes(grain_module: object) -> tuple[type, type]:
    """Dynamically construct Grain classes."""
    base_ds = getattr(grain_module, "RandomAccessDataSource", object)
    base_map = getattr(grain_module, "MapTransform", object)

    class HFDataSource(base_ds):
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

    class MaxTextFormatTransform(base_map):
        """Transforms data into MaxText expected format."""

        def __init__(self, tokenizer: SQLTokenizer) -> None:
            """Execute function."""
            self.tokenizer = tokenizer

        def map(self, element: JSONDict) -> JSONDict:
            """Map an element."""
            prompt = element.get("sql_prompt", element.get("question", ""))
            target = element.get("sql", element.get("query", ""))
            return {"inputs": self.tokenizer.encode(str(prompt)), "targets": self.tokenizer.encode(str(target)), "segment_ids": [1], "positions": [0]}

    return (HFDataSource, MaxTextFormatTransform)


def build_dataloader(dataset_name: str, split: str, batch_size: int = 32, *, distributed: bool = False, tokenizer_name: str | None = None, **kwargs: JSONValue) -> JSONDict:
    """Build a MaxText-specific Grain dataloader."""
    duckdb_path = str(kwargs.get("duckdb_path", ""))
    duckdb_table = str(kwargs.get("duckdb_table", ""))
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

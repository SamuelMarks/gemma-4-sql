"""MLX-specific ETL pipeline."""

from __future__ import annotations

import typing

from gemma_4_sql.tokenization import SQLTokenizer

if typing.TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict, JSONValue

try:
    import datasets
except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
    datasets = None

try:
    import duckdb
except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
    duckdb = None

try:
    import mlx.core as mx
except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
    mx = None


def build_dataloader(dataset_name: str, split: str, batch_size: int = 32, *, distributed: bool = False, tokenizer_name: str | None = None, **kwargs: JSONValue) -> JSONDict:
    """Build an MLX-specific dataloader."""
    duckdb_path = kwargs.get("duckdb_path")
    duckdb_table = kwargs.get("duckdb_table")
    if datasets is None:
        return {"dataset": dataset_name, "split": split, "status": "mocked", "batch_size": batch_size, "backend": "mlx", "distributed": distributed, "mock_samples": [{"query": "SELECT * FROM users", "nl": "Get all users"}]}

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

    tokenizer = SQLTokenizer(model_name=tokenizer_name)

    class MLXDataLoader:
        """Simple DataLoader for MLX that yields padded batches."""

        def __init__(self, ds: object, tok: SQLTokenizer, bs: int) -> None:
            self.ds = ds
            self.tok = tok
            self.bs = bs

        def __iter__(self) -> typing.Iterator[JSONDict]:
            batch_inputs = []
            batch_targets = []
            for item in self.ds:
                prompt = item.get("sql_prompt", item.get("question", ""))
                target = item.get("sql", item.get("query", ""))
                batch_inputs.append(self.tok.encode(prompt))
                batch_targets.append(self.tok.encode(target))

                if len(batch_inputs) == self.bs:
                    max_len_in = max(len(x) for x in batch_inputs)
                    max_len_tgt = max(len(x) for x in batch_targets)

                    padded_in = [x + [0] * (max_len_in - len(x)) for x in batch_inputs]
                    padded_tgt = [x + [0] * (max_len_tgt - len(x)) for x in batch_targets]

                    yield {"inputs": padded_in, "targets": padded_tgt}
                    batch_inputs = []
                    batch_targets = []

            if batch_inputs:
                max_len_in = max(len(x) for x in batch_inputs)
                max_len_tgt = max(len(x) for x in batch_targets)
                padded_in = [x + [0] * (max_len_in - len(x)) for x in batch_inputs]
                padded_tgt = [x + [0] * (max_len_tgt - len(x)) for x in batch_targets]
                yield {"inputs": padded_in, "targets": padded_tgt}

    dataloader = MLXDataLoader(hf_dataset, tokenizer, batch_size)
    return {"dataset": dataset_name, "split": split, "status": "loaded", "batch_size": batch_size, "backend": "mlx", "distributed": distributed, "loader": dataloader}

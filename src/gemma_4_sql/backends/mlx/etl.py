"""MLX-specific ETL pipeline."""

from __future__ import annotations

import typing

from gemma_4_sql.backends.common_data import _load_duckdb_dataset
from gemma_4_sql.backends.lazy_loader import catch_optional_imports
from gemma_4_sql.tokenization import SQLTokenizer
from gemma_4_sql.type_hints import ETLConfig

if typing.TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict, JSONValue
datasets = None
with catch_optional_imports():
    import datasets
duckdb = None
with catch_optional_imports():
    pass
mx = None
with catch_optional_imports():
    pass


def _pad_batch(batch_inputs: list[list[int]], batch_targets: list[list[int]]) -> JSONDict:
    """Pad batch sequences to max length.

    Args:
        batch_inputs: A sequence of batch inputs.
        batch_targets: A sequence of batch targets.

    Returns:
        A dictionary containing the results.
    """
    max_len_in = max(len(x) for x in batch_inputs)
    max_len_tgt = max(len(x) for x in batch_targets)
    padded_in = [x + [0] * (max_len_in - len(x)) for x in batch_inputs]
    padded_tgt = [x + [0] * (max_len_tgt - len(x)) for x in batch_targets]
    return {"inputs": padded_in, "targets": padded_tgt}


class MLXDataLoader:
    """Simple DataLoader for MLX that yields padded batches."""

    def __init__(self, ds: object, tok: SQLTokenizer, bs: int) -> None:
        """Execute logic.

        Args:
            ds: The ds.
            tok: The tok.
            bs: The integer value for bs.
        """
        self.ds = ds
        self.tok = tok
        self.bs = bs

    def __iter__(self) -> typing.Iterator[JSONDict]:
        """Execute logic.

        Yields:
            object: The yielded item during generation.

        """
        batch_inputs = []
        batch_targets = []
        for item in self.ds:
            prompt = item.get("sql_prompt", item.get("question", ""))
            target = item.get("sql", item.get("query", ""))
            batch_inputs.append(self.tok.encode(str(prompt)))
            batch_targets.append(self.tok.encode(str(target)))
            if len(batch_inputs) == self.bs:
                yield _pad_batch(batch_inputs, batch_targets)
                batch_inputs = []
                batch_targets = []
        if batch_inputs:
            yield _pad_batch(batch_inputs, batch_targets)


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


def build_dataloader(config: ETLConfig, **kwargs: JSONValue) -> JSONDict:
    """Build an MLX-specific dataloader.


    Args:
        **kwargs: Overrides for ETL configuration (e.g., duckdb_path, duckdb_table).
    Returns:
        object: The resulting output from the operation.

    """
    dataset_name = config.dataset_name
    split = config.split
    batch_size = config.batch_size
    distributed = config.distributed
    tokenizer_name = config.tokenizer_name
    duckdb_path = str(config.duckdb_path or kwargs.get("duckdb_path") or "")
    duckdb_table = str(config.duckdb_table or kwargs.get("duckdb_table") or "")
    if datasets is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError(f"Missing datasets. Cannot load {dataset_name}.")

    hf_dataset = _load_hf_or_duckdb(dataset_name, split, duckdb_path, duckdb_table)
    tokenizer = SQLTokenizer(model_name=tokenizer_name)
    dataloader = MLXDataLoader(hf_dataset, tokenizer, batch_size)
    return {"dataset": dataset_name, "split": split, "status": "loaded", "batch_size": batch_size, "backend": "mlx", "distributed": distributed, "loader": dataloader}

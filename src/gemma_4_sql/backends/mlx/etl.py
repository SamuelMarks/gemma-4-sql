# Copyright 2024
"""MLX-specific ETL pipeline."""

from __future__ import annotations

import typing

from gemma_4_sql.backends.common_data import _load_duckdb_dataset
from gemma_4_sql.backends.lazy_loader import catch_optional_imports
from gemma_4_sql.tokenization import SQLTokenizer

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

    Returns:
        object: The resulting output from the operation.

    """
    max_len_in = max(len(x) for x in batch_inputs)  # pragma: no cover
    max_len_tgt = max(len(x) for x in batch_targets)  # pragma: no cover
    padded_in = [x + [0] * (max_len_in - len(x)) for x in batch_inputs]  # pragma: no cover
    padded_tgt = [x + [0] * (max_len_tgt - len(x)) for x in batch_targets]  # pragma: no cover
    return {"inputs": padded_in, "targets": padded_tgt}  # pragma: no cover


class MLXDataLoader:
    """Simple DataLoader for MLX that yields padded batches."""

    def __init__(self, ds: object, tok: SQLTokenizer, bs: int) -> None:
        """Execute logic."""
        self.ds = ds  # pragma: no cover
        self.tok = tok  # pragma: no cover
        self.bs = bs  # pragma: no cover

    def __iter__(self) -> typing.Iterator[JSONDict]:
        """Execute logic.

        Yields:
            object: The yielded item during generation.

        """
        batch_inputs = []  # pragma: no cover
        batch_targets = []  # pragma: no cover
        for item in self.ds:  # pragma: no cover
            prompt = item.get("sql_prompt", item.get("question", ""))  # pragma: no cover
            target = item.get("sql", item.get("query", ""))  # pragma: no cover
            batch_inputs.append(self.tok.encode(str(prompt)))  # pragma: no cover
            batch_targets.append(self.tok.encode(str(target)))  # pragma: no cover
            if len(batch_inputs) == self.bs:  # pragma: no cover
                yield _pad_batch(batch_inputs, batch_targets)  # pragma: no cover
                batch_inputs = []  # pragma: no cover
                batch_targets = []  # pragma: no cover
        if batch_inputs:  # pragma: no cover
            yield _pad_batch(batch_inputs, batch_targets)  # pragma: no cover


def build_dataloader(config: object, **kwargs: JSONValue) -> JSONDict:
    """Build an MLX-specific dataloader.

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
    if datasets is None:
        return {"dataset": dataset_name, "split": split, "status": "mocked", "batch_size": batch_size, "backend": "mlx", "distributed": distributed, "mock_samples": [{"query": "SELECT * FROM users", "nl": "Get all users"}]}
    hf_dataset = _load_duckdb_dataset(duckdb_path, duckdb_table) if duckdb_path and duckdb_table else datasets.load_dataset(dataset_name, split=split)  # pragma: no cover
    tokenizer = SQLTokenizer(model_name=tokenizer_name)  # pragma: no cover
    dataloader = MLXDataLoader(hf_dataset, tokenizer, batch_size)  # pragma: no cover
    return {"dataset": dataset_name, "split": split, "status": "loaded", "batch_size": batch_size, "backend": "mlx", "distributed": distributed, "loader": dataloader}  # pragma: no cover

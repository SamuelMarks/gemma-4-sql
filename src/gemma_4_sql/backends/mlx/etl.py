"""MLX-specific ETL pipeline."""

from __future__ import annotations

import typing

from gemma_4_sql.backends.common_data import (
    load_duckdb_dataset,
    load_hf_or_duckdb_dataset,
    pad_token_batch,
)
from gemma_4_sql.backends.lazy_loader import optional_import
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.tokenization import SQLTokenizer

if typing.TYPE_CHECKING:
    from collections.abc import Iterator

    from gemma_4_sql.type_hints import DataloaderResultDict, ETLConfig, JSONDict

datasets = optional_import("datasets")

_pad_batch = pad_token_batch


class MLXDataLoader:
    """Iterable batch dataloader for MLX token sequence training."""

    def __init__(self, ds: object, tok: SQLTokenizer, bs: int) -> None:
        """Initialize the MLXDataLoader.

        Args:
            ds: Underlying Hugging Face Dataset or sequence of SQL prompt/query dictionaries.
            tok: `SQLTokenizer` instance used to encode prompts and SQL targets.
            bs: Number of records per padded batch.

        """
        self.ds = ds
        self.tok = tok
        self.bs = bs

    def __iter__(self) -> Iterator[dict[str, list[list[int]]]]:
        """Iterate over the dataset and yield zero-padded token batches.

        Yields:
            Dictionary containing `'inputs'` and `'targets'` 2D padded integer token lists.

        """
        batch_inputs: list[list[int]] = []
        batch_targets: list[list[int]] = []
        for item in typing.cast("typing.Iterable[JSONDict]", self.ds):
            prompt = str(item.get("sql_prompt", item.get("question", "")))
            target = str(item.get("sql", item.get("query", "")))
            batch_inputs.append(self.tok.encode(prompt))
            batch_targets.append(self.tok.encode(target))
            if len(batch_inputs) == self.bs:
                yield _pad_batch(batch_inputs, batch_targets)
                batch_inputs = []
                batch_targets = []
        if batch_inputs:
            yield _pad_batch(batch_inputs, batch_targets)


def _load_hf_or_duckdb(
    dataset_name: str,
    split: str,
    duckdb_path: str | None,
    duckdb_table: str | None,
) -> object:
    """Load dataset records from DuckDB when configured or Hugging Face `datasets`.

    Args:
        dataset_name: Hugging Face dataset identifier.
        split: Dataset split name (e.g., `'train'`).
        duckdb_path: Optional path to a DuckDB database file.
        duckdb_table: Optional DuckDB table name.

    Returns:
        Loaded dataset sequence or Hugging Face Dataset object.

    Raises:
        DependencyMissingError: If `datasets` is required and not installed.

    """
    return load_hf_or_duckdb_dataset(
        dataset_name=dataset_name,
        split=split,
        duckdb_path=duckdb_path,
        duckdb_table=duckdb_table,
        datasets_module=datasets,
        load_duckdb_fn=load_duckdb_dataset,
    )


def build_dataloader(config: ETLConfig, **kwargs: object) -> DataloaderResultDict:
    """Build an `MLXDataLoader` for the MLX backend.

    Args:
        config: ETL configuration parameters.
        **kwargs: Overrides such as `duckdb_path` and `duckdb_table`.

    Returns:
        Structured `DataloaderResultDict` containing the constructed `MLXDataLoader`.

    Raises:
        DependencyMissingError: If `datasets` is not available.

    """
    dataset_name = config.dataset_name
    split = config.split
    batch_size = config.batch_size
    distributed = config.distributed
    tokenizer_name = config.tokenizer_name
    duckdb_path = str(config.duckdb_path or kwargs.get("duckdb_path") or "")
    duckdb_table = str(config.duckdb_table or kwargs.get("duckdb_table") or "")
    if datasets is None:
        raise DependencyMissingError(f"Missing datasets. Cannot load {dataset_name}.")
    hf_dataset = _load_hf_or_duckdb(dataset_name, split, duckdb_path, duckdb_table)
    tokenizer = SQLTokenizer(model_name=tokenizer_name)
    dataloader = MLXDataLoader(hf_dataset, tokenizer, batch_size)
    return typing.cast(
        "DataloaderResultDict",
        {
            "dataset": dataset_name,
            "split": split,
            "status": "loaded",
            "batch_size": batch_size,
            "backend": "mlx",
            "distributed": distributed,
            "loader": dataloader,
        },
    )

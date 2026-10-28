"""MLX-specific ETL pipeline."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence

from gemma_4_sql.backends.common_data import (
    DatasetSequenceProtocol,
    HFDatasetsModuleProtocol,
    TokenizedElementDict,
    TokenizedElementValue,
    load_duckdb_dataset,
    load_hf_or_duckdb_dataset,
    resolve_duckdb_options,
    tokenize_dataset_element,
)
from gemma_4_sql.backends.common_data import (
    pad_1d_list as _pad_1d_list,
)
from gemma_4_sql.backends.common_data import (
    pad_token_batch as _pad_batch,
)
from gemma_4_sql.backends.lazy_loader import optional_import
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.tokenization import SQLTokenizer
from gemma_4_sql.type_hints import DataloaderResultDict, ETLConfig, JSONDict, JSONValue, TensorType

_raw_datasets = optional_import("datasets")
datasets: HFDatasetsModuleProtocol | None = _raw_datasets if isinstance(_raw_datasets, HFDatasetsModuleProtocol) else None

type MLXCollatedBatchValue = TensorType | Sequence[int] | list[list[int]] | list[list[float]] | list[list[list[float]]] | list[list[list[list[float]]]] | list[TokenizedElementValue]
type MLXCollatedBatchDict = dict[str, MLXCollatedBatchValue]


class MLXDataLoader:
    """Iterable batch dataloader for MLX token sequence and multimodal training."""

    ds: DatasetSequenceProtocol | Iterable[Mapping[str, JSONValue]]
    tok: SQLTokenizer
    bs: int

    def __init__(
        self,
        ds: DatasetSequenceProtocol | Iterable[Mapping[str, JSONValue]],
        tok: SQLTokenizer,
        bs: int,
    ) -> None:
        """Initialize the `MLXDataLoader`.

        Args:
            ds: Underlying Hugging Face Dataset or sequence of SQL prompt/query dictionaries.
            tok: `SQLTokenizer` instance used to encode prompts and SQL targets.
            bs: Number of records per padded batch.

        """
        self.ds = ds
        self.tok = tok
        self.bs = bs

    def _collate_items(self, items: list[TokenizedElementDict]) -> MLXCollatedBatchDict:
        """Pad and stack a batch of tokenized MLX dataset records.

        Args:
            items: List of tokenized item dictionaries.

        Returns:
            Collated batch dictionary containing padded token sequences and optional multimodal arrays.

        """
        batch = _pad_batch(
            [item["inputs"] for item in items],  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]
            [item["targets"] for item in items],  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]
        )
        res: MLXCollatedBatchDict = {
            "inputs": batch["inputs"],
            "targets": batch["targets"],
        }
        if "chosen_inputs" in items[0]:
            res["chosen_inputs"] = _pad_1d_list([item["chosen_inputs"] for item in items])  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]
        if "chosen_labels" in items[0]:
            res["chosen_labels"] = _pad_1d_list([item["chosen_labels"] for item in items])  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]
        if "rejected_inputs" in items[0]:
            res["rejected_inputs"] = _pad_1d_list([item["rejected_inputs"] for item in items])  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]
        if "rejected_labels" in items[0]:
            res["rejected_labels"] = _pad_1d_list([item["rejected_labels"] for item in items])  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]
        if "pixel_values" in items[0]:
            res["pixel_values"] = [item["pixel_values"] for item in items]
        if "audio_values" in items[0]:
            res["audio_values"] = [item["audio_values"] for item in items]
        return res

    def __iter__(self) -> Iterator[Mapping[str, TensorType | Sequence[int]]]:
        """Iterate over the dataset and yield zero-padded token and multimodal batches.

        Yields:
            Dictionary containing `'inputs'`, `'targets'`, and optional preference/multimodal features.

        """
        batch_items: list[TokenizedElementDict] = []
        for item in self.ds:  # pyright: ignore[reportGeneralTypeIssues]
            assert isinstance(item, Mapping)
            tokenized = tokenize_dataset_element(item, self.tok, process_multimodal=True)
            batch_items.append(tokenized)
            if len(batch_items) == self.bs:
                yield self._collate_items(batch_items)  # pyright: ignore[reportReturnType]  # ty: ignore[invalid-yield]
                batch_items = []
        if batch_items:
            yield self._collate_items(batch_items)  # pyright: ignore[reportReturnType]  # ty: ignore[invalid-yield]


def _load_hf_or_duckdb(
    dataset_name: str,
    split: str,
    duckdb_path: str | None,
    duckdb_table: str | None,
) -> DatasetSequenceProtocol | list[JSONDict]:
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


def build_dataloader(config: ETLConfig, **kwargs: JSONValue) -> DataloaderResultDict:
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
    duckdb_path, duckdb_table = resolve_duckdb_options(config, kwargs)
    if datasets is None:
        raise DependencyMissingError(f"Missing datasets. Cannot load {dataset_name}.")
    hf_dataset = _load_hf_or_duckdb(dataset_name, split, duckdb_path, duckdb_table)
    tokenizer = SQLTokenizer(model_name=tokenizer_name)
    dataloader = MLXDataLoader(hf_dataset, tokenizer, batch_size)
    return {
        "dataset": dataset_name,
        "split": split,
        "status": "loaded",
        "batch_size": batch_size,
        "backend": "mlx",
        "distributed": distributed,
        "loader": dataloader,
    }

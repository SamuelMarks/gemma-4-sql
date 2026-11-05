"""MLX-specific ETL pipeline."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import TypeGuard

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


def _is_int_list_seq(val: list[TokenizedElementValue]) -> TypeGuard[list[list[int]]]:
    """Narrow `val` to `list[list[int]]` for token sequence padding.

    Args:
        val: List of tokenized element values.

    Returns:
        `True` for 1D integer token list batches.

    """
    return not isinstance(val, (str, bytes))


def _extract_token_lists(items: list[TokenizedElementDict], key: str) -> list[list[int]]:
    """Extract a list of 1D integer token lists for `key` from `items`.

    Args:
        items: List of tokenized dataset records.
        key: Dictionary key (`'inputs'`, `'targets'`, `'chosen_inputs'`, etc.).

    Returns:
        List of 1D integer token lists.

    """
    raw = [item[key] for item in items]
    assert _is_int_list_seq(raw)
    return raw


def _as_batch_mapping(
    batch: MLXCollatedBatchDict,
) -> TypeGuard[Mapping[str, TensorType | Sequence[int]]]:
    """Narrow `batch` to `Mapping[str, TensorType | Sequence[int]]`.

    Args:
        batch: Collated MLX batch dictionary.

    Returns:
        `True` when `batch` is a non-string mapping.

    """
    return not isinstance(batch, (str, bytes))


def _is_iterable_dataset(
    ds: DatasetSequenceProtocol | Iterable[Mapping[str, JSONValue]],
) -> TypeGuard[Iterable[Mapping[str, JSONValue]]]:
    """Narrow `ds` to `Iterable[Mapping[str, JSONValue]]`.

    Args:
        ds: Dataset sequence or iterable of records.

    Returns:
        `True` when `ds` is not a plain `str`.

    """
    return not isinstance(ds, (str, bytes))


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
            _extract_token_lists(items, "inputs"),
            _extract_token_lists(items, "targets"),
        )
        res: MLXCollatedBatchDict = {
            "inputs": batch["inputs"],
            "targets": batch["targets"],
        }
        if "chosen_inputs" in items[0]:
            res["chosen_inputs"] = _pad_1d_list(_extract_token_lists(items, "chosen_inputs"))
        if "chosen_labels" in items[0]:
            res["chosen_labels"] = _pad_1d_list(_extract_token_lists(items, "chosen_labels"))
        if "rejected_inputs" in items[0]:
            res["rejected_inputs"] = _pad_1d_list(_extract_token_lists(items, "rejected_inputs"))
        if "rejected_labels" in items[0]:
            res["rejected_labels"] = _pad_1d_list(_extract_token_lists(items, "rejected_labels"))
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
        ds_ref = self.ds
        assert _is_iterable_dataset(ds_ref)
        for item in ds_ref:
            assert isinstance(item, Mapping)
            tokenized = tokenize_dataset_element(item, self.tok, process_multimodal=True)
            batch_items.append(tokenized)
            if len(batch_items) == self.bs:
                collated = self._collate_items(batch_items)
                assert _as_batch_mapping(collated)
                yield collated
                batch_items = []
        if batch_items:
            collated_tail = self._collate_items(batch_items)
            assert _as_batch_mapping(collated_tail)
            yield collated_tail


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

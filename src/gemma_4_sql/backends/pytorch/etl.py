"""PyTorch-specific ETL pipeline."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, runtime_checkable

from gemma_4_sql.backends.common_data import (
    DatasetSequenceProtocol,
    HFDatasetsModuleProtocol,
    load_duckdb_dataset,
    load_hf_or_duckdb_dataset,
    resolve_duckdb_options,
    tokenize_dataset_element,
)
from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.tokenization import SQLTokenizer
from gemma_4_sql.type_hints import (
    DataLoaderProtocol,
    DataloaderResultDict,
    ETLConfig,
    JSONDict,
    JSONValue,
    TorchTensorProtocol,
)

type PyTorchItemDict = dict[str, TorchTensorProtocol]


@runtime_checkable
class _DistributedSamplerProtocol(Protocol):
    """Protocol for an initialized `torch.utils.data.distributed.DistributedSampler`."""


@runtime_checkable
class _DistributedSamplerConstructorProtocol(Protocol):
    """Protocol for `torch.utils.data.distributed.DistributedSampler` constructor."""

    def __call__(self, dataset: PyTorchDataset | str, /) -> _DistributedSamplerProtocol:
        """Construct a distributed sampler over `dataset`."""
        ...


@runtime_checkable
class _DataLoaderConstructorProtocol(Protocol):
    """Protocol for `torch.utils.data.DataLoader` constructor."""

    def __call__(
        self,
        dataset: PyTorchDataset,
        *,
        batch_size: int,
        shuffle: bool,
        sampler: _DistributedSamplerProtocol | None,
        collate_fn: Callable[[Sequence[PyTorchItemDict]], PyTorchItemDict],
    ) -> DataLoaderProtocol[PyTorchItemDict]:
        """Construct a PyTorch `DataLoader`."""
        ...


_raw_datasets = optional_import("datasets")
datasets: HFDatasetsModuleProtocol | None = _raw_datasets if isinstance(_raw_datasets, HFDatasetsModuleProtocol) else None
torch = optional_import("torch")
DataLoader: _DataLoaderConstructorProtocol | None = optional_attr(
    "torch.utils.data",
    "DataLoader",
    _DataLoaderConstructorProtocol,
)
Dataset = optional_attr("torch.utils.data", "Dataset")


class PyTorchDataset:
    """Adapter turning a Hugging Face Dataset or sequence of SQL records into a PyTorch map-style dataset."""

    _ds: DatasetSequenceProtocol | list[JSONDict]
    tokenizer: SQLTokenizer

    def __init__(self, hf_ds: DatasetSequenceProtocol | list[JSONDict], tokenizer: SQLTokenizer) -> None:
        """Initialize the `PyTorchDataset`.

        Args:
            hf_ds: Hugging Face Dataset or sequence of SQL records.
            tokenizer: `SQLTokenizer` instance.

        """
        self._ds = hf_ds
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """Return dataset record count."""
        return len(self._ds)

    def __getitem__(self, idx: int) -> PyTorchItemDict:
        """Tokenize and format a single dataset item by index.

        Args:
            idx: Zero-based record index.

        Returns:
            Dictionary of input/target tensors and optional multimodal features.

        """
        assert torch is not None
        element = self._ds[idx]
        assert isinstance(element, Mapping)
        tokenized = tokenize_dataset_element(element, self.tokenizer, process_multimodal=True)
        res: PyTorchItemDict = {
            "inputs": torch.tensor(tokenized["inputs"], dtype=torch.long),
            "targets": torch.tensor(tokenized["targets"], dtype=torch.long),
        }
        if "chosen_inputs" in tokenized:
            res["chosen_inputs"] = torch.tensor(tokenized["chosen_inputs"], dtype=torch.long)
        if "chosen_labels" in tokenized:
            res["chosen_labels"] = torch.tensor(tokenized["chosen_labels"], dtype=torch.long)
        if "rejected_inputs" in tokenized:
            res["rejected_inputs"] = torch.tensor(tokenized["rejected_inputs"], dtype=torch.long)
        if "rejected_labels" in tokenized:
            res["rejected_labels"] = torch.tensor(tokenized["rejected_labels"], dtype=torch.long)
        if "pixel_values" in tokenized:
            res["pixel_values"] = torch.tensor(tokenized["pixel_values"], dtype=torch.float32)
        if "audio_values" in tokenized:
            res["audio_values"] = torch.tensor(tokenized["audio_values"], dtype=torch.float32)
        return res


def _get_pytorch_classes() -> type[PyTorchDataset]:
    """Return the `PyTorchDataset` class.

    Returns:
        The `PyTorchDataset` class.

    """
    return PyTorchDataset


def _collate_fn[TensorT](batch: Sequence[PyTorchItemDict | Mapping[str, TensorT]]) -> PyTorchItemDict:
    """Collate and pad a list of `PyTorchDataset` items into a batch dictionary.

    Args:
        batch: List of item dictionaries from `PyTorchDataset.__getitem__`.

    Returns:
        Collated dictionary with padded `inputs`/`targets` and stacked multimodal tensors.

    """
    assert torch is not None
    inputs = [item["inputs"] for item in batch]
    targets = [item["targets"] for item in batch]
    inputs_padded: TorchTensorProtocol = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    targets_padded: TorchTensorProtocol = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True)
    res: PyTorchItemDict = {"inputs": inputs_padded, "targets": targets_padded}
    for pref_key in ("chosen_inputs", "chosen_labels", "rejected_inputs", "rejected_labels"):
        if all(pref_key in item for item in batch):
            res[pref_key] = torch.nn.utils.rnn.pad_sequence([item[pref_key] for item in batch], batch_first=True)
    if "pixel_values" in batch[0]:
        res["pixel_values"] = torch.stack([item["pixel_values"] for item in batch])
    if "audio_values" in batch[0]:
        res["audio_values"] = torch.stack([item["audio_values"] for item in batch])
    return res


def _get_sampler(source: PyTorchDataset | str, distributed: bool) -> _DistributedSamplerProtocol | None:
    """Return a `DistributedSampler` when `distributed=True`, otherwise `None`.

    Args:
        source: Dataset instance.
        distributed: Whether distributed data sampling is enabled.

    Returns:
        `DistributedSampler` instance or `None`.

    """
    if distributed:
        try:
            sampler_cls = optional_attr(
                "torch.utils.data.distributed",
                "DistributedSampler",
                _DistributedSamplerConstructorProtocol,
            )
            if sampler_cls is not None:
                return sampler_cls(source)
        except (RuntimeError, ValueError, ImportError):
            pass
    return None


def _load_hf_or_duckdb(
    dataset_name: str,
    split: str,
    duckdb_path: str | None,
    duckdb_table: str | None,
) -> DatasetSequenceProtocol | list[JSONDict]:
    """Load dataset from DuckDB or Hugging Face `datasets`.

    Args:
        dataset_name: Hugging Face dataset name.
        split: Dataset split name.
        duckdb_path: Optional path to a DuckDB file.
        duckdb_table: Optional DuckDB table name.

    Returns:
        Loaded dataset object or list of row dictionaries.

    Raises:
        DependencyMissingError: If `datasets` is missing when loading from Hugging Face.

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
    """Build a PyTorch `DataLoader` for the specified dataset.

    Args:
        config: ETL configuration parameters.
        **kwargs: Overrides such as `duckdb_path` and `duckdb_table`.

    Returns:
        Structured `DataloaderResultDict` containing the constructed `DataLoader`.

    Raises:
        DependencyMissingError: If PyTorch or `datasets` dependencies are missing.

    """
    dataset_name = config.dataset_name
    split = config.split
    batch_size = config.batch_size
    distributed = config.distributed
    tokenizer_name = config.tokenizer_name
    duckdb_path, duckdb_table = resolve_duckdb_options(config, kwargs)
    if datasets is None or torch is None or Dataset is None or DataLoader is None:
        raise DependencyMissingError(f"Missing torch or datasets. Cannot load {dataset_name}.")
    hf_dataset = _load_hf_or_duckdb(dataset_name, split, duckdb_path, duckdb_table)
    tokenizer = SQLTokenizer(model_name=tokenizer_name)
    dataset_cls = _get_pytorch_classes()
    source = dataset_cls(hf_dataset, tokenizer=tokenizer)
    sampler = _get_sampler(source, distributed)
    dataloader = DataLoader(source, batch_size=batch_size, shuffle=False, sampler=sampler, collate_fn=_collate_fn)
    return {
        "dataset": dataset_name,
        "split": split,
        "status": "loaded",
        "batch_size": batch_size,
        "backend": "pytorch",
        "distributed": distributed,
        "loader": dataloader,
    }

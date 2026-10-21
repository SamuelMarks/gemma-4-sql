"""PyTorch-specific ETL pipeline."""

from __future__ import annotations

import typing
from typing import Any

from gemma_4_sql.backends.common_data import (
    load_duckdb_dataset,
    load_hf_or_duckdb_dataset,
)
from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.tokenization import SQLTokenizer

if typing.TYPE_CHECKING:
    from gemma_4_sql.type_hints import DataloaderResultDict, ETLConfig, JSONDict

datasets = optional_import("datasets")
torch = optional_import("torch")
DataLoader = optional_attr("torch.utils.data", "DataLoader")
Dataset = optional_attr("torch.utils.data", "Dataset")


def _get_pytorch_classes() -> Any:
    """Dynamically construct the `PyTorchDataset` class inheriting from `torch.utils.data.Dataset`.

    Returns:
        Constructed `PyTorchDataset` class.

    """
    base_cls: Any = Dataset if Dataset is not None else object

    class PyTorchDataset(base_cls):
        """Adapter to turn a Hugging Face Dataset into a PyTorch Dataset."""

        _ds: Any
        tokenizer: SQLTokenizer

        def __init__(self, hf_ds: object, tokenizer: SQLTokenizer) -> None:
            """Initialize the PyTorchDataset.

            Args:
                hf_ds: Hugging Face Dataset or sequence of SQL records.
                tokenizer: `SQLTokenizer` instance.

            """
            self._ds = hf_ds
            self.tokenizer = tokenizer

        def __len__(self) -> int:
            """Return dataset record count."""
            return len(self._ds)

        def __getitem__(self, idx: int) -> JSONDict:
            """Tokenize and format a single dataset item by index.

            Args:
                idx: Zero-based record index.

            Returns:
                Dictionary of input/target tensors and optional multimodal features.

            """
            from gemma_4_sql.backends.common_multimodal import format_multimodal_prompt, process_audio, process_image

            element = self._ds[idx]
            prompt = str(element.get("sql_prompt", element.get("question", "")))
            target = str(element.get("sql", element.get("query", "")))

            image_input = element.get("image_bytes") or element.get("image_url") or element.get("image")
            audio_input = element.get("audio_clip") or element.get("audio")

            formatted = format_multimodal_prompt(
                prompt,
                has_image=image_input is not None,
                has_audio=audio_input is not None,
            )
            prompt_str = str(formatted.get("prompt", prompt))
            res: JSONDict = {
                "inputs": torch.tensor(self.tokenizer.encode(prompt_str), dtype=torch.long),
                "targets": torch.tensor(self.tokenizer.encode(target), dtype=torch.long),
            }
            if image_input is not None:
                img_data = process_image(image_input)
                res["pixel_values"] = torch.tensor(img_data["pixel_values"], dtype=torch.float32)
            if audio_input is not None:
                aud_data = process_audio(audio_input)
                res["audio_values"] = torch.tensor(aud_data["audio_values"], dtype=torch.float32)
            return res

    return PyTorchDataset


def _collate_fn(batch: list[JSONDict]) -> JSONDict:
    """Collate and pad a list of `PyTorchDataset` items into a batch dictionary.

    Args:
        batch: List of item dictionaries from `PyTorchDataset.__getitem__`.

    Returns:
        Collated dictionary with padded `inputs`/`targets` and stacked multimodal tensors.

    """
    inputs = [item["inputs"] for item in batch]
    targets = [item["targets"] for item in batch]
    inputs_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    targets_padded = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True)
    res: JSONDict = {"inputs": inputs_padded, "targets": targets_padded}
    if "pixel_values" in batch[0]:
        res["pixel_values"] = torch.stack([item["pixel_values"] for item in batch])
    if "audio_values" in batch[0]:
        res["audio_values"] = torch.stack([item["audio_values"] for item in batch])
    return res


def _get_sampler(source: object, distributed: bool) -> object:
    """Return a `DistributedSampler` when `distributed=True`, otherwise `None`.

    Args:
        source: Dataset instance.
        distributed: Whether distributed data sampling is enabled.

    Returns:
        `DistributedSampler` instance or `None`.

    """
    if distributed:
        try:
            sampler_cls = optional_attr("torch.utils.data.distributed", "DistributedSampler")
            if sampler_cls is not None:
                return sampler_cls(source)
        except (RuntimeError, ValueError, ImportError):
            pass
    return None


def _load_hf_or_duckdb(dataset_name: str, split: str, duckdb_path: str | None, duckdb_table: str | None) -> object:
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


def build_dataloader(config: ETLConfig, **kwargs: object) -> DataloaderResultDict:
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
    duckdb_path = str(config.duckdb_path or kwargs.get("duckdb_path") or "")
    duckdb_table = str(config.duckdb_table or kwargs.get("duckdb_table") or "")
    if datasets is None or torch is None or Dataset is None or DataLoader is None:
        raise DependencyMissingError(f"Missing torch or datasets. Cannot load {dataset_name}.")
    hf_dataset = _load_hf_or_duckdb(dataset_name, split, duckdb_path, duckdb_table)
    tokenizer = SQLTokenizer(model_name=tokenizer_name)
    PyTorchDataset = _get_pytorch_classes()
    source = PyTorchDataset(hf_dataset, tokenizer=tokenizer)
    sampler = _get_sampler(source, distributed)
    dataloader = DataLoader(source, batch_size=batch_size, shuffle=False, sampler=sampler, collate_fn=_collate_fn)
    return typing.cast(
        "DataloaderResultDict",
        {
            "dataset": dataset_name,
            "split": split,
            "status": "loaded",
            "batch_size": batch_size,
            "backend": "pytorch",
            "distributed": distributed,
            "loader": dataloader,
        },
    )

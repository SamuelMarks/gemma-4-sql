"""PyTorch-specific ETL pipeline."""

from __future__ import annotations

import typing
from typing import Any

from gemma_4_sql.backends.common_data import _load_duckdb_dataset
from gemma_4_sql.tokenization import SQLTokenizer
from gemma_4_sql.type_hints import ETLConfig, JSONDict

if typing.TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONValue

try:
    import datasets as _datasets
    import torch as _torch
    from torch.utils.data import DataLoader as _DataLoader
    from torch.utils.data import Dataset as _Dataset

    datasets: typing.Any = _datasets
    torch: typing.Any = _torch
    DataLoader: typing.Any = _DataLoader
    Dataset: typing.Any = _Dataset
except (ImportError, AttributeError):
    datasets = None
    torch = None
    DataLoader = None
    Dataset = None
duckdb = None


def _get_pytorch_classes() -> type:
    """Dynamically construct PyTorch Dataset class.

    Returns:
        The execution result.
    """

    class PyTorchDataset(Dataset if Dataset is not None else object):  # type: ignore[misc]
        """PyTorch Dataset wrapping Hugging Face."""

        _ds: typing.Any

        def __init__(self, hf_ds: object, tok: SQLTokenizer) -> None:
            """Execute the load duckdb dataset operation."""
            self._ds = hf_ds
            self._tok = tok

        def __len__(self) -> int:
            """Return the total length.

            Returns:
                The total number of samples.
            """
            return len(self._ds)

        def __getitem__(self, idx: int) -> JSONDict:
            """Retrieve an item by its index with multimodal feature extraction.

            Args:
                idx: Sample index in dataset.

            Returns:
                Dictionary containing tokenized inputs, targets, and optional pixel/audio tensors.
            """
            from gemma_4_sql.backends.common_multimodal import (
                format_multimodal_prompt,
                process_audio,
                process_image,
            )

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
            item: dict[str, Any] = {
                "inputs": torch.tensor(self._tok.encode(str(formatted["prompt"])), dtype=torch.long),
                "targets": torch.tensor(self._tok.encode(str(target)), dtype=torch.long),
            }
            if image_input is not None:
                img_data = process_image(image_input)
                item["pixel_values"] = torch.tensor(img_data["pixel_values"], dtype=torch.float32)
            if audio_input is not None:
                aud_data = process_audio(audio_input)
                item["audio_values"] = torch.tensor(aud_data["audio_values"], dtype=torch.float32)
            return item

    return PyTorchDataset


def _collate_fn(batch: list[JSONDict]) -> JSONDict:
    """Collate individual items into batched tensors.

    Args:
        batch: List of dataset elements.

    Returns:
        Dictionary of batched PyTorch tensors.
    """
    inputs = [item["inputs"] for item in batch]
    targets = [item["targets"] for item in batch]
    inputs_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    targets_padded = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True)
    res: dict[str, Any] = {"inputs": inputs_padded, "targets": targets_padded}
    if "pixel_values" in batch[0]:
        res["pixel_values"] = torch.stack([item["pixel_values"] for item in batch])
    if "audio_values" in batch[0]:
        res["audio_values"] = torch.stack([item["audio_values"] for item in batch])
    return res


def _get_sampler(pt_dataset: typing.Any, distributed: bool) -> object:
    """Get the appropriate PyTorch sampler for data loading.

    Args:
        pt_dataset: The PyTorch dataset.
        distributed: Whether to use a DistributedSampler.

    Returns:
        A DistributedSampler if distributed is true, otherwise None.
    """
    if not distributed:
        return None
    distributed_sampler_cls = __import__("torch.utils.data.distributed", fromlist=["DistributedSampler"]).DistributedSampler
    try:
        return distributed_sampler_cls(pt_dataset)
    except (RuntimeError, ValueError):
        return None


def _load_hf_or_duckdb(dataset_name: str, split: str, duckdb_path: str | None, duckdb_table: str | None) -> object:
    """Load a dataset from Hugging Face or DuckDB.

    Args:
        dataset_name: The name of the Hugging Face dataset.
        split: The dataset split to load.
        duckdb_path: Optional path to a DuckDB database.
        duckdb_table: Optional name of the DuckDB table.

    Returns:
        The loaded dataset.

    Raises:
        DependencyMissingError: If datasets dependency is missing.
    """
    if duckdb_path and duckdb_table:
        return _load_duckdb_dataset(duckdb_path, duckdb_table)
    if datasets is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("Datasets dependency is missing.")
    return datasets.load_dataset(dataset_name, split=split)


def build_dataloader(config: ETLConfig, **kwargs: JSONValue) -> JSONDict:
    """Build a PyTorch-specific dataloader.

    Args:
        config: ETL configuration parameters.
        **kwargs: Overrides for ETL configuration (e.g., duckdb_path, duckdb_table).

    Returns:
        A dictionary containing the PyTorch DataLoader and metadata.

    Raises:
        DependencyMissingError: If required dependencies are missing.
    """
    dataset_name = config.dataset_name
    split = config.split
    batch_size = config.batch_size
    distributed = config.distributed
    tokenizer_name = config.tokenizer_name
    duckdb_path = str(config.duckdb_path or kwargs.get("duckdb_path") or "")
    duckdb_table = str(config.duckdb_table or kwargs.get("duckdb_table") or "")
    if datasets is None or torch is None or Dataset is None or (DataLoader is None):
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError(f"Missing PyTorch or datasets. Cannot load {dataset_name}.")

    hf_dataset = _load_hf_or_duckdb(dataset_name, split, duckdb_path, duckdb_table)
    tokenizer = SQLTokenizer(model_name=tokenizer_name)
    pt_dataset_cls = _get_pytorch_classes()
    pt_dataset = pt_dataset_cls(hf_dataset, tokenizer)
    sampler = _get_sampler(pt_dataset, distributed)
    dataloader = DataLoader(pt_dataset, batch_size=batch_size, shuffle=sampler is None, sampler=sampler, collate_fn=_collate_fn)
    return typing.cast(JSONDict, {"dataset": dataset_name, "split": split, "status": "loaded", "batch_size": batch_size, "backend": "pytorch", "distributed": distributed, "loader": dataloader})

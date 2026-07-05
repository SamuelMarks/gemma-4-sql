# Copyright 2024
"""PyTorch-specific ETL pipeline."""

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
torch = None
DataLoader = None
Dataset = None
with catch_optional_imports():
    import torch
    from torch.utils.data import DataLoader, Dataset
duckdb = None
with catch_optional_imports():
    pass


def _get_pytorch_classes() -> type:
    """Dynamically construct PyTorch Dataset class.

    Returns:
        object: The resulting output from the operation.

    """

    class PyTorchDataset(Dataset):
        """PyTorch Dataset wrapping Hugging Face."""

        def __init__(self, hf_ds: object, tok: SQLTokenizer) -> None:
            """Execute the load duckdb dataset operation."""
            self._ds = hf_ds
            self._tok = tok

        def __len__(self) -> int:
            """Return the total length.

            Returns:
                object: The resulting output from the operation.

            """
            return len(self._ds)

        def __getitem__(self, idx: int) -> JSONDict:
            """Retrieve an item by its index.

            Returns:
                object: The resulting output from the operation.

            """
            element = self._ds[idx]
            prompt = element.get("sql_prompt", element.get("question", ""))
            target = element.get("sql", element.get("query", ""))
            return {"inputs": torch.tensor(self._tok.encode(str(prompt)), dtype=torch.long), "targets": torch.tensor(self._tok.encode(str(target)), dtype=torch.long)}

    return PyTorchDataset


def _collate_fn(batch: list[JSONDict]) -> JSONDict:
    """Collate batches.

    Returns:
        object: The resulting output from the operation.

    """
    inputs = [item["inputs"] for item in batch]
    targets = [item["targets"] for item in batch]
    inputs_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    targets_padded = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True)
    return {"inputs": inputs_padded, "targets": targets_padded}


def _get_sampler(pt_dataset: Dataset, distributed: bool) -> object:
    if not distributed:
        return None
    distributed_sampler_cls = __import__("torch.utils.data.distributed", fromlist=["DistributedSampler"]).DistributedSampler
    try:
        return distributed_sampler_cls(pt_dataset)
    except (RuntimeError, ValueError):  # pragma: no cover
        return None  # pragma: no cover


def _load_hf_or_duckdb(dataset_name: str, split: str, duckdb_path: str | None, duckdb_table: str | None) -> object:
    if duckdb_path and duckdb_table:
        return _load_duckdb_dataset(duckdb_path, duckdb_table)
    return datasets.load_dataset(dataset_name, split=split)


def build_dataloader(config: ETLConfig, **kwargs: JSONValue) -> JSONDict:
    """Build a PyTorch-specific dataloader.

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
    if datasets is None or torch is None or Dataset is None or (DataLoader is None):
        return {"dataset": dataset_name, "split": split, "status": "mocked", "batch_size": batch_size, "backend": "pytorch", "distributed": distributed, "mock_samples": [{"query": "SELECT * FROM users", "nl": "Get all users"}]}
    hf_dataset = _load_hf_or_duckdb(dataset_name, split, duckdb_path, duckdb_table)
    tokenizer = SQLTokenizer(model_name=tokenizer_name)
    pt_dataset_cls = _get_pytorch_classes()
    pt_dataset = pt_dataset_cls(hf_dataset, tokenizer)
    sampler = _get_sampler(pt_dataset, distributed)
    dataloader = DataLoader(pt_dataset, batch_size=batch_size, shuffle=sampler is None, sampler=sampler, collate_fn=_collate_fn)
    return {"dataset": dataset_name, "split": split, "status": "loaded", "batch_size": batch_size, "backend": "pytorch", "distributed": distributed, "loader": dataloader}

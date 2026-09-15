"""PyTorch-specific ETL pipeline."""

from __future__ import annotations

import typing

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

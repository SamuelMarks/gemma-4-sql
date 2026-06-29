"""PyTorch-specific ETL pipeline."""

from __future__ import annotations

import typing

from gemma_4_sql.tokenization import SQLTokenizer

try:
    import datasets
except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
    datasets = None
try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
    torch = None
    DataLoader = None
    Dataset = None
try:
    import duckdb
except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
    duckdb = None


def build_dataloader(dataset_name: str, split: str, batch_size: int = 32, *, distributed: bool = False, tokenizer_name: str | None = None, **kwargs: object) -> dict[str, object]:
    """Build a PyTorch-specific dataloader."""
    duckdb_path = kwargs.get("duckdb_path")
    duckdb_table = kwargs.get("duckdb_table")
    if datasets is None or torch is None or Dataset is None or (DataLoader is None):
        return {"dataset": dataset_name, "split": split, "status": "mocked", "batch_size": batch_size, "backend": "pytorch", "distributed": distributed, "mock_samples": [{"query": "SELECT * FROM users", "nl": "Get all users"}]}
    if duckdb_path and duckdb_table:
        if duckdb is None:
            msg = "duckdb is required for DuckDB support."
            raise ImportError(msg)
        conn = duckdb.connect(duckdb_path, read_only=True)
        try:
            hf_dataset = conn.execute("SELECT * FROM ?", (duckdb_table,)).fetchdf().to_dict(orient="records")
        finally:
            conn.close()
    else:
        hf_dataset = datasets.load_dataset(dataset_name, split=split)
    tokenizer = SQLTokenizer(model_name=tokenizer_name)

    class PyTorchDataset(Dataset):  # type: ignore[misc]
        """PyTorch Dataset wrapping Hugging Face."""

        def __init__(self: typing.Any, hf_ds: object, tok: SQLTokenizer) -> None:
            """Initialize with dataset and tokenizer."""
            self._ds = hf_ds
            self._tok = tok

        def __len__(self: typing.Any) -> int:
            """Return dataset length."""
            return len(self._ds)

        def __getitem__(self: typing.Any, idx: int) -> dict[str, object]:
            """Get and format dataset item."""
            element = self._ds[idx]
            prompt = element.get("sql_prompt", element.get("question", ""))
            target = element.get("sql", element.get("query", ""))
            return {"inputs": torch.tensor(self._tok.encode(prompt), dtype=torch.long), "targets": torch.tensor(self._tok.encode(target), dtype=torch.long)}

    pt_dataset = PyTorchDataset(hf_dataset, tokenizer)

    def collate_fn(batch: list[dict[str, object]]) -> dict[str, object]:
        """Collate batches."""
        inputs = [item["inputs"] for item in batch]
        targets = [item["targets"] for item in batch]
        inputs_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
        targets_padded = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True)
        return {"inputs": inputs_padded, "targets": targets_padded}

    dataloader = DataLoader(pt_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return {"dataset": dataset_name, "split": split, "status": "loaded", "batch_size": batch_size, "backend": "pytorch", "distributed": distributed, "loader": dataloader}

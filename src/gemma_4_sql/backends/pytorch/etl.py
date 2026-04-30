"""
PyTorch-specific ETL pipeline.
"""

from __future__ import annotations

from typing import Any

from gemma_4_sql.tokenization import SQLTokenizer

try:
    import datasets
except ImportError:
    datasets = None

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    torch = None
    DataLoader = None
    Dataset = None

try:
    import duckdb
except ImportError:
    duckdb = None

def build_dataloader(
    dataset_name: str,
    split: str,
    batch_size: int = 32,
    distributed: bool = False,
    tokenizer_name: str | None = None,
    duckdb_path: str | None = None,
    duckdb_table: str | None = None,
) -> dict[str, Any]:
    """Builds a PyTorch-specific dataloader."""
    if datasets is None or torch is None or Dataset is None or DataLoader is None:
        return {
            "dataset": dataset_name,
            "split": split,
            "status": "mocked",
            "batch_size": batch_size,
            "backend": "pytorch",
            "distributed": distributed,
            "mock_samples": [{"query": "SELECT * FROM users", "nl": "Get all users"}],
        }

    if duckdb_path and duckdb_table:
        if duckdb is None:
            raise ImportError("duckdb is required for DuckDB support.")
        conn = duckdb.connect(duckdb_path, read_only=True)
        try:
            hf_dataset = conn.execute(f"SELECT * FROM {duckdb_table}").fetchdf().to_dict(orient="records")
        finally:
            conn.close()
    else:
        hf_dataset = datasets.load_dataset(dataset_name, split=split)
    tokenizer = SQLTokenizer(model_name=tokenizer_name)

    class PyTorchDataset(Dataset):
        """PyTorch Dataset wrapping Hugging Face."""

        def __init__(self, hf_ds: Any, tok: SQLTokenizer):
            """Initialize with dataset and tokenizer."""
            self._ds = hf_ds
            self._tok = tok

        def __len__(self) -> int:
            """Return dataset length."""
            return len(self._ds)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            """Get and format dataset item."""
            element = self._ds[idx]
            prompt = element.get("sql_prompt", element.get("question", ""))
            target = element.get("sql", element.get("query", ""))
            return {
                "inputs": torch.tensor(self._tok.encode(prompt), dtype=torch.long),
                "targets": torch.tensor(self._tok.encode(target), dtype=torch.long),
            }

    pt_dataset = PyTorchDataset(hf_dataset, tokenizer)

    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate batches."""
        inputs = [item["inputs"] for item in batch]
        targets = [item["targets"] for item in batch]

        # Pad to max length in batch
        inputs_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
        targets_padded = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True)

        return {"inputs": inputs_padded, "targets": targets_padded}

    dataloader = DataLoader(
        pt_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return {
        "dataset": dataset_name,
        "split": split,
        "status": "loaded",
        "batch_size": batch_size,
        "backend": "pytorch",
        "distributed": distributed,
        "loader": dataloader,
    }

"""
ETL module for generating SQL training datasets using grain.
"""

from __future__ import annotations

from typing import Any


def _route_backend(
    dataset_name: str,
    split: str,
    batch_size: int,
    backend: str,
    distributed: bool = False,
    tokenizer_name: str | None = None,
    duckdb_path: str | None = None,
    duckdb_table: str | None = None,
) -> dict[str, Any]:
    """
    Routes the ETL request to the specific backend implementation.

    Args:
        dataset_name: The dataset to load.
        split: The dataset split to load.
        batch_size: Batch size for Grain DataLoader.
        backend: The backend ecosystem ('jax', 'keras', or 'maxtext').
        distributed: Whether to use distributed sharding.
        tokenizer_name: Optional Hugging Face tokenizer name.
        duckdb_path: Optional path to DuckDB database for data loading.
        duckdb_table: Optional DuckDB table or query to load data from.

    Returns:
        Dict containing dataset metadata and loader status.
    """
    kwargs = {
        "dataset_name": dataset_name,
        "split": split,
        "batch_size": batch_size,
        "distributed": distributed,
        "tokenizer_name": tokenizer_name,
        "duckdb_path": duckdb_path,
        "duckdb_table": duckdb_table,
    }

    from gemma_4_sql.sdk.registry import get_backend
    return get_backend(backend).build_dataloader(**kwargs)


def etl_pretrain(
    dataset_name: str = "seeklhy/SynSQL-2.5M",
    split: str = "train",
    batch_size: int = 32,
    backend: str = "jax",
    distributed: bool = False,
    tokenizer_name: str | None = None,
    duckdb_path: str | None = None,
    duckdb_table: str | None = None,
) -> dict[str, Any]:
    """
    ETL pipeline for pretraining SQL datasets.

    Args:
        dataset_name: The Hugging Face dataset identifier. Defaults to SynSQL-2.5M.
        split: The dataset split.
        batch_size: Batch size for dataloader.
        backend: The target ecosystem ('jax', 'keras', or 'maxtext').
        distributed: Whether to use distributed sharding.
        tokenizer_name: Optional Hugging Face tokenizer name.
        duckdb_path: Optional path to DuckDB database.
        duckdb_table: Optional DuckDB table name.

    Returns:
        A dictionary containing metadata and dataset representation.
    """
    return _route_backend(
        dataset_name,
        split,
        batch_size,
        backend,
        distributed,
        tokenizer_name,
        duckdb_path,
        duckdb_table,
    )


def etl_sft(
    dataset_name: str = "gretelai/synthetic_text_to_sql",
    split: str = "train",
    batch_size: int = 32,
    backend: str = "jax",
    distributed: bool = False,
    tokenizer_name: str | None = None,
    duckdb_path: str | None = None,
    duckdb_table: str | None = None,
) -> dict[str, Any]:
    """
    ETL pipeline for SFT (Supervised Fine-Tuning) SQL datasets.

    Args:
        dataset_name: The Hugging Face dataset identifier.
        split: The dataset split.
        batch_size: Batch size for dataloader.
        backend: The target ecosystem ('jax', 'keras', or 'maxtext').
        distributed: Whether to use distributed sharding.
        tokenizer_name: Optional Hugging Face tokenizer name.
        duckdb_path: Optional path to DuckDB database.
        duckdb_table: Optional DuckDB table name.

    Returns:
        A dictionary containing metadata and dataset representation.
    """
    return _route_backend(
        dataset_name,
        split,
        batch_size,
        backend,
        distributed,
        tokenizer_name,
        duckdb_path,
        duckdb_table,
    )


def etl_posttrain(
    dataset_name: str = "xlangai/spider2-lite",
    split: str = "train",
    batch_size: int = 32,
    backend: str = "jax",
    distributed: bool = False,
    tokenizer_name: str | None = None,
    duckdb_path: str | None = None,
    duckdb_table: str | None = None,
) -> dict[str, Any]:
    """
    ETL pipeline for post-training/RLHF SQL datasets.

    Args:
        dataset_name: The Hugging Face dataset identifier.
        split: The dataset split.
        batch_size: Batch size for dataloader.
        backend: The target ecosystem ('jax', 'keras', or 'maxtext').
        distributed: Whether to use distributed sharding.
        tokenizer_name: Optional Hugging Face tokenizer name.
        duckdb_path: Optional path to DuckDB database.
        duckdb_table: Optional DuckDB table name.

    Returns:
        A dictionary containing metadata and dataset representation.
    """
    return _route_backend(
        dataset_name,
        split,
        batch_size,
        backend,
        distributed,
        tokenizer_name,
        duckdb_path,
        duckdb_table,
    )

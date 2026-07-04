# Copyright 2024
"""ETL module for generating SQL training datasets using grain."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemma_4_sql.constants import DEFAULT_POSTTRAIN_DATASET, DEFAULT_SFT_DATASET
from gemma_4_sql.type_hints import ETLConfig

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict, JSONValue


def _route_backend(config: ETLConfig, backend: str, **kwargs: JSONValue) -> JSONDict:
    """Routes the ETL request to the specific backend implementation.

    Args:
    ----
        config: The ETL configuration.
        backend: The backend ecosystem ('jax', 'keras', or 'maxtext').
        **kwargs: Additional keyword arguments.

    Returns:
    -------
        Dict containing dataset metadata and loader status.

    """
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    return get_backend(backend).build_dataloader(config, **kwargs)


def etl_pretrain(config: ETLConfig | None = None, backend: str = "jax") -> JSONDict:
    """ETL pipeline for pretraining SQL datasets.

    Args:
    ----
        config: The ETL configuration.
        backend: The target ecosystem ('jax', 'keras', or 'maxtext').

    Returns:
    -------
        A dictionary containing metadata and dataset representation.

    """
    if config is None:
        config = ETLConfig(dataset_name="seeklhy/SynSQL-2.5M", split="train")
    return _route_backend(config, backend)


def etl_sft(config: ETLConfig | None = None, backend: str = "jax") -> JSONDict:
    """ETL pipeline for SFT (Supervised Fine-Tuning) SQL datasets.

    Args:
    ----
        config: The ETL configuration.
        backend: The target ecosystem ('jax', 'keras', or 'maxtext').

    Returns:
    -------
        A dictionary containing metadata and dataset representation.

    """
    if config is None:
        config = ETLConfig(dataset_name=DEFAULT_SFT_DATASET, split="train")
    return _route_backend(config, backend)


def etl_posttrain(config: ETLConfig | None = None, backend: str = "jax") -> JSONDict:
    """ETL pipeline for post-training/RLHF SQL datasets.

    Args:
    ----
        config: The ETL configuration.
        backend: The target ecosystem ('jax', 'keras', or 'maxtext').

    Returns:
    -------
        A dictionary containing metadata and dataset representation.

    """
    if config is None:
        config = ETLConfig(dataset_name=DEFAULT_POSTTRAIN_DATASET, split="train")
    return _route_backend(config, backend)

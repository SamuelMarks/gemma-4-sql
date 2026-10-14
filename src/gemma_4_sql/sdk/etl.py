"""ETL module for generating SQL training datasets using grain."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from gemma_4_sql.constants import DEFAULT_BACKEND, DEFAULT_POSTTRAIN_DATASET, DEFAULT_PRETRAIN_DATASET, DEFAULT_SFT_DATASET
from gemma_4_sql.sdk.registry import get_backend
from gemma_4_sql.type_hints import ETLConfig

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import ETLResultDict, JSONValue


def _route_backend(config: ETLConfig, backend: str, **kwargs: JSONValue) -> ETLResultDict:
    """Route the ETL request to the specific backend implementation.

    Args:
        config: The configuration parameters.
        backend: The backend framework to use.
        **kwargs: Additional keyword arguments passed to the specific backend implementation.

    Returns:
        A dictionary containing the results.

    """
    return get_backend(backend).build_dataloader(config, **kwargs)


def etl_pretrain(config: ETLConfig | None = None, backend: str = DEFAULT_BACKEND) -> ETLResultDict:
    """Run ETL pipeline for pretraining SQL datasets.

    Args:
        config: The ETL configuration.
        backend: The target ecosystem ('jax', 'keras', 'maxtext', 'mlx', or 'pytorch').

    Returns:
        A dictionary containing metadata and dataset representation.

    """
    if config is None:
        cfg = ETLConfig(dataset_name=DEFAULT_PRETRAIN_DATASET, split="train", stage="pretrain")
    else:
        cfg = copy.copy(config)
        cfg.stage = "pretrain"
    return _route_backend(cfg, backend)


def etl_sft(config: ETLConfig | None = None, backend: str = DEFAULT_BACKEND) -> ETLResultDict:
    """Run ETL pipeline for SFT (Supervised Fine-Tuning) SQL datasets.

    Args:
        config: The ETL configuration.
        backend: The target ecosystem ('jax', 'keras', 'maxtext', 'mlx', or 'pytorch').

    Returns:
        A dictionary containing metadata and dataset representation.

    """
    if config is None:
        cfg = ETLConfig(dataset_name=DEFAULT_SFT_DATASET, split="train", stage="sft")
    else:
        cfg = copy.copy(config)
        cfg.stage = "sft"
    return _route_backend(cfg, backend)


def etl_posttrain(config: ETLConfig | None = None, backend: str = DEFAULT_BACKEND) -> ETLResultDict:
    """Run ETL pipeline for post-training/RLHF SQL datasets.

    Args:
        config: The ETL configuration.
        backend: The target ecosystem ('jax', 'keras', 'maxtext', 'mlx', or 'pytorch').

    Returns:
        A dictionary containing metadata and dataset representation.

    """
    if config is None:
        cfg = ETLConfig(dataset_name=DEFAULT_POSTTRAIN_DATASET, split="train", stage="posttrain")
    else:
        cfg = copy.copy(config)
        cfg.stage = "posttrain"
    return _route_backend(cfg, backend)


__all__ = ["etl_posttrain", "etl_pretrain", "etl_sft"]

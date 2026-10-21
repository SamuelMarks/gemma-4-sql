"""JAX-specific Grain ETL pipeline."""

from __future__ import annotations

import typing

from gemma_4_sql.backends.common_data import (
    build_grain_dataloader,
    get_grain_classes,
    get_grain_sampler,
    load_duckdb_dataset,
    load_hf_or_duckdb_dataset,
)
from gemma_4_sql.backends.lazy_loader import optional_import
from gemma_4_sql.tokenization import SQLTokenizer
from gemma_4_sql.type_hints import ETLConfig

if typing.TYPE_CHECKING:
    from gemma_4_sql.type_hints import ETLResultDict, JSONValue

datasets: typing.Any = optional_import("datasets")
grain: typing.Any = optional_import("grain.python")


def _load_hf_or_duckdb(
    dataset_name: str,
    split: str,
    duckdb_path: str | None,
    duckdb_table: str | None,
) -> object:
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
    return load_hf_or_duckdb_dataset(
        dataset_name,
        split,
        duckdb_path,
        duckdb_table,
        datasets,
        load_duckdb_fn=load_duckdb_dataset,
    )


def _get_sampler(source_len: int, distributed: bool) -> object:
    """Provide the appropriate Grain sampler for data loading.

    Args:
        source_len: The length of the source data.
        distributed: Whether to use distributed sharding.

    Returns:
        A Grain IndexSampler.

    Raises:
        DependencyMissingError: If Grain dependency is missing.

    """
    return get_grain_sampler(source_len, distributed, grain)


def build_dataloader(config: ETLConfig, **kwargs: JSONValue) -> ETLResultDict:
    """Build a JAX-specific Grain dataloader.

    Args:
        config: The configuration parameters.
        **kwargs: Overrides for ETL configuration (e.g., duckdb_path, duckdb_table).

    Returns:
        A dictionary containing the results.

    Raises:
        DependencyMissingError: If grain or datasets are missing.

    """
    return build_grain_dataloader(
        config,
        backend_name="jax",
        datasets_module=datasets,
        grain_module=grain,
        load_fn=_load_hf_or_duckdb,
        grain_classes_fn=get_grain_classes,
        tokenizer_cls=SQLTokenizer,
        sampler_fn=_get_sampler,
        **kwargs,
    )

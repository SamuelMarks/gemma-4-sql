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
        grain_classes_fn=get_grain_classes,
        tokenizer_cls=SQLTokenizer,
        **kwargs,
    )


__all__ = [
    "SQLTokenizer",
    "build_dataloader",
    "datasets",
    "get_grain_classes",
    "get_grain_sampler",
    "grain",
    "load_duckdb_dataset",
    "load_hf_or_duckdb_dataset",
]

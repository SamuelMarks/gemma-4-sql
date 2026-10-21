"""MaxText-specific Grain ETL pipeline."""

from __future__ import annotations

import typing
from typing import Any

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
    from gemma_4_sql.type_hints import ETLResultDict, JSONDict, JSONValue

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


def _wrap_maxtext_transform(base_transform_cls: Any) -> Any:
    """Wrap the base Grain transform class to inject MaxText `segment_ids` and `positions`.

    Args:
        base_transform_cls: Base transform class from `get_grain_classes`.

    Returns:
        `MaxTextFormatTransform` subclass.

    """

    class MaxTextFormatTransform(base_transform_cls):
        """Transforms data into MaxText expected format with segment_ids and positions."""

        def map(self, element: JSONDict) -> JSONDict:
            """Map an element with MaxText specific features.

            Args:
                element: Input dataset element dictionary.

            Returns:
                Tokenized dictionary augmented with `segment_ids` and `positions`.

            """
            result: dict[str, Any] = super().map(element)
            result["segment_ids"] = [1]
            result["positions"] = [0]
            return result

    return MaxTextFormatTransform


def build_dataloader(config: ETLConfig, **kwargs: JSONValue) -> ETLResultDict:
    """Build a MaxText-specific Grain dataloader.

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
        backend_name="maxtext",
        datasets_module=datasets,
        grain_module=grain,
        load_fn=_load_hf_or_duckdb,
        grain_classes_fn=get_grain_classes,
        tokenizer_cls=SQLTokenizer,
        sampler_fn=_get_sampler,
        transform_wrapper=_wrap_maxtext_transform,
        **kwargs,
    )

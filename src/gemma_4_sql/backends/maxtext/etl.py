"""MaxText-specific Grain ETL pipeline."""

from __future__ import annotations

from gemma_4_sql.backends.common_data import (
    GrainModuleProtocol,
    HFDatasetsModuleProtocol,
    build_grain_dataloader,
    get_grain_classes,
    get_grain_sampler,
    load_duckdb_dataset,
    load_hf_or_duckdb_dataset,
    wrap_maxtext_transform,
)
from gemma_4_sql.backends.lazy_loader import optional_import
from gemma_4_sql.tokenization import SQLTokenizer
from gemma_4_sql.type_hints import ETLConfig, ETLResultDict, JSONValue

_raw_datasets = optional_import("datasets")
datasets: HFDatasetsModuleProtocol | None = _raw_datasets if isinstance(_raw_datasets, HFDatasetsModuleProtocol) else None
_raw_grain = optional_import("grain.python")
grain: GrainModuleProtocol | None = _raw_grain if isinstance(_raw_grain, GrainModuleProtocol) else None
_wrap_maxtext_transform = wrap_maxtext_transform


def build_dataloader(config: ETLConfig, **kwargs: JSONValue) -> ETLResultDict:
    """Build a MaxText-specific Grain dataloader.

    Args:
        config: The configuration parameters.
        **kwargs: Overrides for ETL configuration (e.g., `duckdb_path`, `duckdb_table`).

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
        grain_classes_fn=get_grain_classes,
        tokenizer_cls=SQLTokenizer,
        transform_wrapper=_wrap_maxtext_transform,  # ty: ignore[invalid-argument-type]
        extra_options=kwargs,
    )


__all__ = [
    "SQLTokenizer",
    "_wrap_maxtext_transform",
    "build_dataloader",
    "datasets",
    "get_grain_classes",
    "get_grain_sampler",
    "grain",
    "load_duckdb_dataset",
    "load_hf_or_duckdb_dataset",
]

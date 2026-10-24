"""Shared dataset, Grain transformation, and DuckDB loading utilities."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from gemma_4_sql.backends.lazy_loader import LazyLoader
from gemma_4_sql.exceptions import DependencyMissingError, ETLError
from gemma_4_sql.tokenization import SQLTokenizer

if TYPE_CHECKING:
    from collections.abc import Callable

    from gemma_4_sql.type_hints import ETLConfig, ETLResultDict, JSONDict

logger = logging.getLogger(__name__)


def _create_hf_data_source(base_ds: Any) -> Any:
    """Create a Grain RandomAccessDataSource subclass wrapping a Hugging Face dataset.

    Args:
        base_ds: Base class (`grain.RandomAccessDataSource` or `object`).

    Returns:
        Constructed `HFDataSource` class.

    """

    class HFDataSource(base_ds):
        """Adapter to turn Hugging Face Dataset into Grain RandomAccessDataSource."""

        _ds: Any

        def __init__(self, hf_ds: Any) -> None:
            """Initialize data source with underlying Hugging Face dataset.

            Args:
                hf_ds: Hugging Face Dataset or sequence of records.

            """
            self._ds = hf_ds

        def __len__(self) -> int:
            """Return total number of records in the dataset.

            Returns:
                Dataset record count.

            """
            return len(self._ds)

        def __getitem__(self, idx: int) -> object:
            """Retrieve a single dataset element by index.

            Args:
                idx: Zero-based record index.

            Returns:
                Dataset record at `idx`.

            """
            return self._ds[idx]

    return HFDataSource


def tokenize_dataset_element(
    element: JSONDict,
    tokenizer: SQLTokenizer,
    *,
    process_multimodal: bool = False,
) -> dict[str, Any]:
    """Extract and tokenize prompt, SQL target, and optional multimodal inputs from a dataset element.

    Args:
        element: Input dictionary element from dataset.
        tokenizer: `SQLTokenizer` instance.
        process_multimodal: Whether to format multimodal placeholders and extract image/audio tensors.

    Returns:
        Dictionary containing `'inputs'`, `'targets'`, and optional `'pixel_values'` / `'audio_values'`.

    """
    prompt = str(element.get("sql_prompt", element.get("question", "")))
    target = str(element.get("sql", element.get("query", "")))

    if not process_multimodal:
        return {
            "inputs": tokenizer.encode(prompt),
            "targets": tokenizer.encode(target),
        }

    from gemma_4_sql.backends.common_multimodal import (
        format_multimodal_prompt,
        process_audio,
        process_image,
    )

    image_input = element.get("image_bytes") or element.get("image_url") or element.get("image")
    audio_input = element.get("audio_clip") or element.get("audio")

    formatted = format_multimodal_prompt(
        prompt,
        has_image=image_input is not None,
        has_audio=audio_input is not None,
    )
    res: dict[str, Any] = {
        "inputs": tokenizer.encode(str(formatted.get("prompt", prompt))),
        "targets": tokenizer.encode(str(target)),
    }
    if image_input is not None:
        img_data = process_image(image_input)
        res["pixel_values"] = img_data["pixel_values"]
    if audio_input is not None:
        aud_data = process_audio(audio_input)
        res["audio_values"] = aud_data["audio_values"]
    return res


def wrap_maxtext_transform(base_transform_cls: Any) -> Any:
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


def _create_base_format_transform(base_map: Any) -> Any:
    """Create a Grain MapTransform subclass for SQL tokenization and multimodal preprocessing.

    Args:
        base_map: Base class (`grain.MapTransform` or `object`).

    Returns:
        Constructed `BaseFormatTransform` class.

    """

    class BaseFormatTransform(base_map):
        """Transforms dataset records into tokenized and multimodal array dictionaries."""

        def __init__(self, tokenizer: SQLTokenizer) -> None:
            """Initialize transform with tokenizer.

            Args:
                tokenizer: The SQL tokenizer instance.

            """
            self.tokenizer = tokenizer

        def map(self, element: JSONDict) -> JSONDict:
            """Execute data mapping transformation with multimodal support.

            Args:
                element: Input dictionary element from dataset.

            Returns:
                Dictionary with tokenized inputs, targets, and optional multimodal features.

            """
            return tokenize_dataset_element(element, self.tokenizer, process_multimodal=True)

    return BaseFormatTransform


def get_grain_classes(grain_module: object) -> tuple[Any, Any]:
    """Dynamically construct Grain data source and transform classes.

    Args:
        grain_module: The loaded `grain.python` module.

    Returns:
        A tuple of `(HFDataSource, BaseFormatTransform)` classes.

    """
    base_ds = getattr(grain_module, "RandomAccessDataSource", object)
    base_map = getattr(grain_module, "MapTransform", object)

    return (_create_hf_data_source(base_ds), _create_base_format_transform(base_map))


def load_hf_or_duckdb_dataset(
    dataset_name: str,
    split: str,
    duckdb_path: str | None,
    duckdb_table: str | None,
    datasets_module: Any,
    load_duckdb_fn: Callable[[str, str], list[JSONDict]] | None = None,
) -> object:
    """Load a dataset from DuckDB if configured, otherwise from Hugging Face `datasets`.

    Args:
        dataset_name: Name of the Hugging Face dataset.
        split: Dataset split to load.
        duckdb_path: Optional path to a DuckDB database file.
        duckdb_table: Optional DuckDB table name.
        datasets_module: Loaded `datasets` module or `None`.
        load_duckdb_fn: Optional override callable for DuckDB dataset loading.

    Returns:
        The loaded dataset object or list of records.

    Raises:
        DependencyMissingError: If `datasets_module` is `None` when loading from Hugging Face.

    """
    duckdb_loader = load_duckdb_fn or load_duckdb_dataset
    if duckdb_path and duckdb_table:
        return duckdb_loader(duckdb_path, duckdb_table)
    if datasets_module is None:
        raise DependencyMissingError("Datasets dependency is missing.", dependency_name="datasets")
    return datasets_module.load_dataset(dataset_name, split=split)


def get_grain_sampler(source_len: int, distributed: bool, grain_module: Any) -> object:
    """Construct a Grain `IndexSampler` with optional JAX distributed sharding.

    Args:
        source_len: Number of records in the data source.
        distributed: Whether to enable `JAXDistributedSharding`.
        grain_module: Loaded `grain.python` module or `None`.

    Returns:
        Configured `grain.IndexSampler` instance.

    Raises:
        DependencyMissingError: If `grain_module` is `None`.

    """
    if grain_module is None:
        raise DependencyMissingError("Grain dependency is missing.", dependency_name="grain")
    shard_options = getattr(grain_module, "JAXDistributedSharding", lambda: None)() if distributed else getattr(grain_module, "NoSharding", lambda: None)()
    return grain_module.IndexSampler(
        num_records=source_len,
        shard_options=shard_options,
        shuffle=False,
        num_epochs=1,
    )


def build_grain_dataloader(
    config: ETLConfig,
    backend_name: str,
    datasets_module: Any,
    grain_module: Any,
    load_fn: Any = None,
    grain_classes_fn: Any = None,
    tokenizer_cls: Any = SQLTokenizer,
    sampler_fn: Any = None,
    transform_wrapper: Any = None,
    **kwargs: Any,
) -> ETLResultDict:
    """Build a Grain DataLoader pipeline shared across JAX, Keras, and MaxText backends.

    Args:
        config: ETL configuration parameters.
        backend_name: Target backend identifier ('jax', 'keras', 'maxtext').
        datasets_module: Loaded Hugging Face `datasets` module.
        grain_module: Loaded `grain.python` module.
        load_fn: Optional dataset loader override.
        grain_classes_fn: Optional `get_grain_classes` override.
        tokenizer_cls: Tokenizer class to instantiate.
        sampler_fn: Optional sampler factory override.
        transform_wrapper: Optional function wrapping `base_transform_cls` (used by MaxText).
        **kwargs: Overrides for `duckdb_path` or `duckdb_table`.

    Returns:
        Dictionary containing the constructed Grain `DataLoader` and metadata.

    Raises:
        DependencyMissingError: If `datasets_module` or `grain_module` is `None`.

    """
    dataset_name = config.dataset_name
    split = config.split
    batch_size = config.batch_size
    distributed = config.distributed
    tokenizer_name = config.tokenizer_name
    duckdb_path = str(config.duckdb_path or kwargs.get("duckdb_path") or "")
    duckdb_table = str(config.duckdb_table or kwargs.get("duckdb_table") or "")

    if datasets_module is None or grain_module is None:
        raise DependencyMissingError(
            f"Missing grain or datasets. Cannot load {dataset_name}.",
            backend=backend_name,
        )

    def _default_load(ds: str, sp: str, dp: str | None, dt: str | None) -> object:
        """Load dataset using default Hugging Face or DuckDB loader."""
        return load_hf_or_duckdb_dataset(ds, sp, dp, dt, datasets_module)

    def _default_sampler(length: int, dist: bool) -> object:
        """Create default Grain sampler for dataset length and distributed setting."""
        return get_grain_sampler(length, dist, grain_module)

    effective_load = load_fn or _default_load
    effective_classes = grain_classes_fn or get_grain_classes
    effective_sampler = sampler_fn or _default_sampler

    hf_dataset = effective_load(dataset_name, split, duckdb_path, duckdb_table)
    data_source_cls, transform_cls = effective_classes(grain_module)
    if transform_wrapper is not None:
        transform_cls = transform_wrapper(transform_cls)
    source = data_source_cls(hf_dataset)
    tokenizer = tokenizer_cls(model_name=tokenizer_name)
    sampler = effective_sampler(len(source), distributed)
    dataloader = grain_module.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=[transform_cls(tokenizer=tokenizer), grain_module.Batch(batch_size=batch_size)],
    )
    return {
        "dataset": dataset_name,
        "split": split,
        "status": "loaded",
        "batch_size": batch_size,
        "backend": backend_name,
        "distributed": distributed,
        "loader": dataloader,
    }


def pad_token_batch(inputs: list[list[int]], targets: list[list[int]]) -> dict[str, list[list[int]]]:
    """Pad variable-length input and target token sequences to uniform batch lengths with zeros.

    Args:
        inputs: Batch of input token ID sequences.
        targets: Batch of target token ID sequences.

    Returns:
        Dictionary with `'inputs'` and `'targets'` keys mapped to zero-padded 2D token lists.

    """
    max_in = max((len(x) for x in inputs), default=1)
    max_tgt = max((len(x) for x in targets), default=1)
    padded_in = [x + [0] * (max_in - len(x)) for x in inputs]
    padded_tgt = [x + [0] * (max_tgt - len(x)) for x in targets]
    return {"inputs": padded_in, "targets": padded_tgt}


def load_duckdb_dataset(db_path: str, table: str) -> list[JSONDict]:
    """Load a dataset from a DuckDB database.

    Args:
        db_path: Path to DuckDB database file.
        table: Table name to read from.

    Returns:
        A list of dictionaries representing the dataset.

    Raises:
        DependencyMissingError: If DuckDB is not available.
        ETLError: If the table name is invalid or DuckDB query execution fails.

    """
    duckdb_module: Any = LazyLoader("duckdb").get_module()
    if duckdb_module is None:
        msg = "duckdb is required. Install with `pip install duckdb`."
        raise DependencyMissingError(msg, dependency_name="duckdb")

    raw_err = getattr(duckdb_module, "Error", None)
    duckdb_error: type[Exception] = raw_err if isinstance(raw_err, type) and issubclass(raw_err, Exception) else RuntimeError
    conn: Any = None
    try:
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", table):
            msg = f"Invalid or unsafe table name: {table!r}"
            raise ValueError(msg)
        conn = duckdb_module.connect(db_path, read_only=True)
        results: list[tuple[Any, ...]] = conn.execute(f'SELECT * FROM "{table}"').fetchall()
        default_cols = [("col" + str(i),) for i in range(len(results[0]))] if results else []
        desc_list: list[tuple[str, ...]] = getattr(conn, "description", default_cols)
        columns = [desc[0] for desc in desc_list]
        return [dict(zip(columns, row)) for row in results]
    except (duckdb_error, ValueError, TypeError, OSError, RuntimeError, AttributeError) as e:
        logger.exception("Failed to load dataset from DuckDB")
        msg = f"DuckDB error: {e}"
        raise ETLError(msg, cause=e) from e
    finally:
        if conn is not None and hasattr(conn, "close"):
            conn.close()

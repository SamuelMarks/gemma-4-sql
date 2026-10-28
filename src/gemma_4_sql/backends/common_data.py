"""Shared dataset, Grain transformation, and DuckDB loading utilities."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, TypedDict, runtime_checkable

from gemma_4_sql.exceptions import DependencyMissingError, ETLError, StatusValue
from gemma_4_sql.tokenization import SQLTokenizer
from gemma_4_sql.type_hints import (
    AudioInput,
    DataLoaderProtocol,
    ETLConfig,
    ETLResultDict,
    ImageInput,
    JSONDict,
    JSONPrimitive,
    JSONValue,
    NDArrayProtocol,
    TensorType,
    TokenizerProtocol,
)

logger = logging.getLogger(__name__)


class RawSQLDatasetRow(TypedDict, total=False):
    """Typed schema for a raw Text-to-SQL, DPO, or multimodal dataset record."""

    sql_prompt: str
    question: str
    prompt: str
    sql: str
    query: str
    chosen: str
    chosen_sql: str
    rejected: str
    rejected_sql: str
    image_bytes: ImageInput
    image_url: ImageInput
    image: ImageInput
    audio_clip: AudioInput
    audio: AudioInput


type TokenizedElementValue = list[int] | NDArrayProtocol | list[float] | JSONValue
type TokenizedElementDict = dict[str, TokenizedElementValue]


@runtime_checkable
class DatasetSequenceProtocol(Protocol):
    """Protocol for an indexable sequence of dataset records."""

    def __len__(self) -> int:
        """Return the number of records in the dataset."""
        ...

    def __getitem__(self, idx: int, /) -> JSONDict | Mapping[str, JSONValue] | int | str:
        """Retrieve a dataset record by zero-based index."""
        ...


@runtime_checkable
class HFDatasetsModuleProtocol(Protocol):
    """Protocol for the Hugging Face `datasets` module."""

    def load_dataset(self, path: str, *, split: str = "train") -> DatasetSequenceProtocol:
        """Load a dataset split by name."""
        ...


@runtime_checkable
class GrainShardingCallableProtocol(Protocol):
    """Protocol for a Grain sharding class or factory."""

    def __call__(self) -> JSONValue:
        """Instantiate sharding options."""
        ...


@runtime_checkable
class GrainModuleWithShardingProtocol(Protocol):
    """Protocol for a `grain.python` module exposing `JAXDistributedSharding` and `NoSharding`."""

    @property
    def JAXDistributedSharding(self) -> GrainShardingCallableProtocol:
        """Provide the distributed sharding constructor."""
        ...

    @property
    def NoSharding(self) -> GrainShardingCallableProtocol:
        """Provide the single-process sharding constructor."""
        ...


@runtime_checkable
class GrainModuleClassesProtocol(Protocol):
    """Protocol for a `grain.python` module exposing `RandomAccessDataSource` and `MapTransform`."""

    @property
    def RandomAccessDataSource(self) -> type:
        """Provide the base class for random access data sources."""
        ...

    @property
    def MapTransform(self) -> type:
        """Provide the base class for elementwise map transforms."""
        ...


@runtime_checkable
class GrainModuleProtocol(GrainModuleWithShardingProtocol, GrainModuleClassesProtocol, Protocol):
    """Protocol for the `grain.python` module."""

    def IndexSampler(
        self,
        *,
        num_records: int,
        shard_options: JSONValue,
        shuffle: bool = False,
        num_epochs: int = 1,
    ) -> JSONValue:
        """Create an `IndexSampler`."""
        ...

    def Batch(self, *, batch_size: int) -> JSONValue:
        """Create a `Batch` operation."""
        ...

    def DataLoader(
        self,
        *,
        data_source: DatasetSequenceProtocol,
        sampler: JSONValue,
        operations: Sequence[_DefaultBaseTransform | JSONValue],
    ) -> DataLoaderProtocol[Mapping[str, TensorType | Sequence[int]]]:
        """Construct a `DataLoader`."""
        ...


@runtime_checkable
class DuckDBDescribedCursorProtocol(Protocol):
    """Protocol for a DuckDB cursor or connection with `.description` and `.fetchall()`."""

    @property
    def description(self) -> Sequence[tuple[str, ...]] | None:
        """Provide column metadata tuples."""
        ...

    def fetchall(self) -> list[tuple[JSONPrimitive, ...]]:
        """Fetch all result rows."""
        ...


@runtime_checkable
class DuckDBRawConnectionProtocol(Protocol):
    """Protocol for a raw DuckDB connection."""

    @property
    def description(self) -> Sequence[tuple[str, ...]] | None:
        """Provide column metadata tuples on the connection."""
        ...

    def execute(self, query: str) -> DuckDBDescribedCursorProtocol:
        """Execute a SQL query and return a cursor."""
        ...

    def close(self) -> None:
        """Close the DuckDB connection."""
        ...


@runtime_checkable
class DuckDBModuleWithErrorProtocol(Protocol):
    """Protocol for the `duckdb` module with `.Error` and `.connect(...)`."""

    @property
    def Error(self) -> type | str | None:
        """Provide the DuckDB base exception class."""
        ...

    def connect(self, database: str, *, read_only: bool = False) -> DuckDBRawConnectionProtocol:
        """Open a DuckDB database connection."""
        ...


@runtime_checkable
class DuckDBConnectCallableProtocol(Protocol):
    """Protocol for a `duckdb` module exposing `.connect(...)`."""

    def connect(self, database: str, *, read_only: bool = False) -> DuckDBRawConnectionProtocol:
        """Open a DuckDB database connection."""
        ...


class _DefaultBaseSource:
    """Default base class for `HFDataSource` when Grain is not installed."""

    _ds: DatasetSequenceProtocol

    def __init__(self, hf_ds: DatasetSequenceProtocol) -> None:
        """Initialize data source with underlying dataset."""
        self._ds = hf_ds

    def __len__(self) -> int:
        """Return total number of records in the dataset."""
        return len(self._ds)

    def __getitem__(self, idx: int) -> JSONDict | Mapping[str, JSONValue] | int | str:
        """Retrieve a single dataset element by index."""
        return self._ds[idx]


class _DefaultBaseTransform:
    """Default base class for `BaseFormatTransform` when Grain is not installed."""


def _create_hf_data_source(base_ds: type) -> type[_DefaultBaseSource]:
    """Create a Grain RandomAccessDataSource subclass wrapping a Hugging Face dataset.

    Args:
        base_ds: Base class (`grain.RandomAccessDataSource` or `_DefaultBaseSource`).

    Returns:
        Constructed `HFDataSource` class.

    """
    with contextlib.suppress(TypeError):
        bases = (base_ds,) if issubclass(base_ds, _DefaultBaseSource) else (_DefaultBaseSource, base_ds)
        cls: type[_DefaultBaseSource] = type(
            "HFDataSource",
            bases,
            {"__doc__": "Adapter to turn Hugging Face Dataset into Grain RandomAccessDataSource."},
        )
        return cls
    return type(
        "HFDataSource",
        (_DefaultBaseSource,),
        {"__doc__": "Adapter to turn Hugging Face Dataset into Grain RandomAccessDataSource."},
    )


def resolve_duckdb_options(config: ETLConfig, kwargs: Mapping[str, StatusValue]) -> tuple[str | None, str | None]:
    """Resolve optional `duckdb_path` and `duckdb_table` strings from `ETLConfig` and `kwargs`.

    Args:
        config: `ETLConfig` instance.
        kwargs: Keyword argument overrides.

    Returns:
        Tuple of `(duckdb_path, duckdb_table)` where unset values are `None`.

    """
    raw_dp = config.duckdb_path or kwargs.get("duckdb_path")
    raw_dt = config.duckdb_table or kwargs.get("duckdb_table")
    duckdb_path: str | None = str(raw_dp) if raw_dp else None
    duckdb_table: str | None = str(raw_dt) if raw_dt else None
    return (duckdb_path, duckdb_table)


def tokenize_dataset_element(
    element: Mapping[str, JSONValue | ImageInput | AudioInput] | RawSQLDatasetRow,
    tokenizer: TokenizerProtocol,
    *,
    process_multimodal: bool = False,
) -> TokenizedElementDict:
    """Extract and tokenize prompt, SQL target, preference pairs, and optional multimodal inputs from a dataset element.

    Args:
        element: Input dictionary element from dataset.
        tokenizer: `SQLTokenizer` or `TokenizerProtocol` instance.
        process_multimodal: Whether to format multimodal placeholders and extract image/audio tensors.

    Returns:
        Dictionary containing `'inputs'`, `'targets'`, optional `'chosen_inputs'`, `'chosen_labels'`,
        `'rejected_inputs'`, `'rejected_labels'`, and optional `'pixel_values'` / `'audio_values'`.

    """
    prompt = str(element.get("sql_prompt", element.get("question", element.get("prompt", ""))))
    target = str(element.get("sql", element.get("query", element.get("chosen", ""))))
    chosen_raw = element.get("chosen", element.get("chosen_sql"))
    rejected_raw = element.get("rejected", element.get("rejected_sql"))

    if not process_multimodal:
        base_res: TokenizedElementDict = {
            "inputs": tokenizer.encode(prompt),
            "targets": tokenizer.encode(target),
        }
        if chosen_raw is not None or rejected_raw is not None:
            chosen_str = str(chosen_raw if chosen_raw is not None else target)
            rejected_str = str(rejected_raw if rejected_raw is not None else f"{target} -- rejected")
            nl = chr(10)
            base_res["chosen_inputs"] = tokenizer.encode(f"{prompt}{nl}{chosen_str}".strip())
            base_res["chosen_labels"] = tokenizer.encode(chosen_str)
            base_res["rejected_inputs"] = tokenizer.encode(f"{prompt}{nl}{rejected_str}".strip())
            base_res["rejected_labels"] = tokenizer.encode(rejected_str)
        return base_res

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
    resolved_prompt = str(formatted.get("prompt", prompt))
    res: TokenizedElementDict = {
        "inputs": tokenizer.encode(resolved_prompt),
        "targets": tokenizer.encode(str(target)),
    }
    if chosen_raw is not None or rejected_raw is not None:
        chosen_str = str(chosen_raw if chosen_raw is not None else target)
        rejected_str = str(rejected_raw if rejected_raw is not None else f"{target} -- rejected")
        nl = chr(10)
        res["chosen_inputs"] = tokenizer.encode(f"{resolved_prompt}{nl}{chosen_str}".strip())
        res["chosen_labels"] = tokenizer.encode(chosen_str)
        res["rejected_inputs"] = tokenizer.encode(f"{resolved_prompt}{nl}{rejected_str}".strip())
        res["rejected_labels"] = tokenizer.encode(rejected_str)
    if image_input is not None and isinstance(image_input, (str, bytes, NDArrayProtocol)):
        img_data = process_image(image_input)
        res["pixel_values"] = img_data["pixel_values"]
    if audio_input is not None and isinstance(audio_input, (str, bytes, list, tuple, NDArrayProtocol)):
        aud_data = process_audio(audio_input)  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]
        res["audio_values"] = aud_data["audio_values"]
    return res


class _BaseFormatTransformRoot(_DefaultBaseTransform):
    """Root class for format transforms with a typed `.map(element)` method."""

    tokenizer: TokenizerProtocol

    def __init__(self, tokenizer: TokenizerProtocol) -> None:
        """Initialize transform with tokenizer.

        Args:
            tokenizer: The SQL tokenizer instance.

        """
        self.tokenizer = tokenizer

    def map(self, element: Mapping[str, JSONValue | ImageInput | AudioInput] | RawSQLDatasetRow) -> TokenizedElementDict:
        """Execute data mapping transformation with multimodal support.

        Args:
            element: Input dictionary element from dataset.

        Returns:
            Dictionary with tokenized inputs, targets, and optional multimodal features.

        """
        return tokenize_dataset_element(element, self.tokenizer, process_multimodal=True)


def wrap_maxtext_transform(base_transform_cls: type[_BaseFormatTransformRoot]) -> type[_BaseFormatTransformRoot]:
    """Wrap the base Grain transform class to inject MaxText `segment_ids` and `positions`.

    Args:
        base_transform_cls: Base transform class from `get_grain_classes`.

    Returns:
        `MaxTextFormatTransform` subclass.

    """

    class MaxTextFormatTransform(_BaseFormatTransformRoot):
        """Transforms data into MaxText expected format with segment_ids and positions."""

        def __init__(self, tokenizer: TokenizerProtocol) -> None:
            """Initialize MaxText transform with wrapped base transform."""
            super().__init__(tokenizer)
            self._base = base_transform_cls(tokenizer)

        def map(self, element: Mapping[str, JSONValue | ImageInput | AudioInput] | RawSQLDatasetRow) -> TokenizedElementDict:
            """Map an element with MaxText specific features.

            Args:
                element: Input dataset element dictionary.

            Returns:
                Tokenized dictionary augmented with `segment_ids` and `positions`.

            """
            result: TokenizedElementDict = self._base.map(element)
            inputs_seq = result.get("inputs", [])
            seq_len = max(1, len(inputs_seq)) if isinstance(inputs_seq, list) else 1
            result["segment_ids"] = [1] * seq_len
            result["positions"] = list(range(seq_len))
            return result

    return MaxTextFormatTransform


def _create_base_format_transform(base_map: type) -> type[_BaseFormatTransformRoot]:
    """Create a Grain MapTransform subclass for SQL tokenization and multimodal preprocessing.

    Args:
        base_map: Base class (`grain.MapTransform` or `_DefaultBaseTransform`).

    Returns:
        Constructed `BaseFormatTransform` class.

    """
    with contextlib.suppress(TypeError):
        bases = (base_map,) if issubclass(base_map, _BaseFormatTransformRoot) else (_BaseFormatTransformRoot, base_map)
        cls: type[_BaseFormatTransformRoot] = type(
            "BaseFormatTransform",
            bases,
            {"__doc__": "Transforms dataset records into tokenized and multimodal array dictionaries."},
        )
        return cls
    return type(
        "BaseFormatTransform",
        (_BaseFormatTransformRoot,),
        {"__doc__": "Transforms dataset records into tokenized and multimodal array dictionaries."},
    )


def get_grain_classes(grain_module: GrainModuleClassesProtocol | None) -> tuple[type[_DefaultBaseSource], type[_BaseFormatTransformRoot]]:
    """Dynamically construct Grain data source and transform classes.

    Args:
        grain_module: The loaded `grain.python` module.

    Returns:
        A tuple of `(HFDataSource, BaseFormatTransform)` classes.

    """
    base_ds: type = _DefaultBaseSource
    base_map: type = _DefaultBaseTransform
    if grain_module is not None:
        with contextlib.suppress(AttributeError):
            base_ds = grain_module.RandomAccessDataSource
        with contextlib.suppress(AttributeError):
            base_map = grain_module.MapTransform

    return (_create_hf_data_source(base_ds), _create_base_format_transform(base_map))


def load_hf_or_duckdb_dataset(
    dataset_name: str,
    split: str,
    duckdb_path: str | None,
    duckdb_table: str | None,
    datasets_module: HFDatasetsModuleProtocol | None,
    load_duckdb_fn: Callable[[str, str], list[JSONDict]] | None = None,
) -> DatasetSequenceProtocol | list[JSONDict]:
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


def get_grain_sampler(
    source_len: int,
    distributed: bool,
    grain_module: GrainModuleProtocol | None,
) -> JSONValue:
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
    shard_fn = grain_module.JAXDistributedSharding if distributed else grain_module.NoSharding
    shard_options: JSONValue = shard_fn() if callable(shard_fn) else None
    return grain_module.IndexSampler(
        num_records=source_len,
        shard_options=shard_options,
        shuffle=False,
        num_epochs=1,
    )


def build_grain_dataloader(
    config: ETLConfig,
    backend_name: str,
    datasets_module: HFDatasetsModuleProtocol | None,
    grain_module: GrainModuleProtocol | None,
    load_fn: Callable[[str, str, str | None, str | None], DatasetSequenceProtocol | list[JSONDict]] | None = None,
    grain_classes_fn: Callable[[GrainModuleProtocol | None], tuple[type, type]] | None = None,
    tokenizer_cls: Callable[..., TokenizerProtocol] = SQLTokenizer,
    sampler_fn: Callable[[int, bool], JSONValue] | None = None,
    transform_wrapper: Callable[[type], type] | None = None,
    extra_options: Mapping[str, JSONValue] | None = None,
    **kwargs: StatusValue,
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
        extra_options: Optional mapping of runtime overrides (such as `duckdb_path`, `duckdb_table`).
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
    merged_options: dict[str, JSONValue] = dict(extra_options) if extra_options is not None else {}
    merged_options.update(kwargs)
    duckdb_path, duckdb_table = resolve_duckdb_options(config, merged_options)

    if datasets_module is None or grain_module is None:
        raise DependencyMissingError(
            f"Missing grain or datasets. Cannot load {dataset_name}.",
            backend=backend_name,
        )

    def _default_load(ds: str, sp: str, dp: str | None, dt: str | None) -> DatasetSequenceProtocol | list[JSONDict]:
        """Load dataset using default Hugging Face or DuckDB loader."""
        return load_hf_or_duckdb_dataset(ds, sp, dp, dt, datasets_module)

    def _default_sampler(length: int, dist: bool) -> JSONValue:
        """Create default Grain sampler for dataset length and distributed setting."""
        return get_grain_sampler(length, dist, grain_module)

    effective_load = load_fn or _default_load
    effective_classes: Callable[[GrainModuleProtocol | None], tuple[type, type]] = grain_classes_fn if grain_classes_fn is not None else get_grain_classes
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


def pad_1d_list(seqs: list[list[int]]) -> list[list[int]]:
    """Zero-pad a list of 1D integer token lists to the maximum sequence length in the batch.

    Args:
        seqs: List of variable-length integer token sequences.

    Returns:
        Zero-padded 2D integer token list.

    """
    max_len = max((len(x) for x in seqs), default=1)
    return [x + [0] * (max_len - len(x)) for x in seqs]


def pad_token_batch(inputs: list[list[int]], targets: list[list[int]]) -> dict[str, list[list[int]]]:
    """Pad variable-length input and target token sequences to uniform batch lengths with zeros.

    Args:
        inputs: Batch of input token ID sequences.
        targets: Batch of target token ID sequences.

    Returns:
        Dictionary with `'inputs'` and `'targets'` keys mapped to zero-padded 2D token lists.

    """
    return {"inputs": pad_1d_list(inputs), "targets": pad_1d_list(targets)}


def _extract_column_descriptions(
    conn: DuckDBRawConnectionProtocol,
    cur: DuckDBDescribedCursorProtocol,
    default_cols: list[tuple[str, ...]],
) -> list[tuple[str, ...]]:
    """Extract column description tuples from `conn` or `cur`, falling back to `default_cols`."""
    desc = None
    with contextlib.suppress(AttributeError):
        desc = conn.description
    if not isinstance(desc, (list, tuple)) or not desc:
        with contextlib.suppress(AttributeError):
            desc = cur.description
    return list(desc) if isinstance(desc, (list, tuple)) and desc else default_cols


def load_duckdb_dataset(db_path: str, table: str) -> list[JSONDict]:
    """Load a dataset from a DuckDB database using `DuckDBAdapter`.

    Args:
        db_path: Path to DuckDB database file.
        table: Table name to read from.

    Returns:
        A list of dictionaries representing the dataset.

    Raises:
        DependencyMissingError: If DuckDB is not available.
        ETLError: If the table name is invalid or DuckDB query execution fails.

    """
    from gemma_4_sql.backends.lazy_loader import LazyLoader
    from gemma_4_sql.exceptions import DatabaseExecutionError
    from gemma_4_sql.sdk.sql_lexer import validate_sql_identifier

    duckdb_module = LazyLoader("duckdb").get_module()
    if duckdb_module is None or not isinstance(duckdb_module, DuckDBConnectCallableProtocol):
        msg = "duckdb is required. Install with `pip install duckdb`."
        raise DependencyMissingError(msg, dependency_name="duckdb")

    duckdb_error: type[Exception] = RuntimeError
    with contextlib.suppress(AttributeError):
        raw_err = duckdb_module.Error  # pyright: ignore[reportAttributeAccessIssue]
        duckdb_error = raw_err if isinstance(raw_err, type) and issubclass(raw_err, Exception) else RuntimeError

    raw_conn: DuckDBRawConnectionProtocol | None = None
    try:
        validated_table = validate_sql_identifier(table)
        raw_conn = duckdb_module.connect(db_path, read_only=True)
        cur = raw_conn.execute(f'SELECT * FROM "{validated_table}"')
        results: Sequence[Sequence[JSONPrimitive]] = cur.fetchall()
        default_cols: list[tuple[str, ...]] = [("col" + str(i),) for i in range(len(results[0]))] if results else []
        desc_list = _extract_column_descriptions(raw_conn, cur, default_cols)
        columns = [desc[0] for desc in desc_list]
        return [dict(zip(columns, row)) for row in results]
    except (duckdb_error, DatabaseExecutionError, ValueError, TypeError, OSError, RuntimeError, AttributeError) as e:
        logger.exception("Failed to load dataset from DuckDB")
        msg = f"DuckDB error: {e}"
        raise ETLError(msg, cause=e) from e
    finally:
        if raw_conn is not None:
            raw_conn.close()

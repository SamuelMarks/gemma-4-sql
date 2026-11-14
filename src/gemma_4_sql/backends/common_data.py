"""Shared dataset, Grain transformation, and DuckDB loading utilities."""

from __future__ import annotations

import abc
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Protocol, TypedDict, runtime_checkable

from gemma_4_sql.backends.lazy_loader import LazyLoader, SubmoduleNamespaceProtocol
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
_DUCKDB_LOADER = LazyLoader("duckdb")


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


class GrainShardingOptionsProtocol(Protocol):
    """Opaque protocol for Grain `ShardOptions` instances (`NoSharding`, `ShardByJaxProcess`, or test descriptors)."""


class GrainSamplerProtocol(Protocol):
    """Opaque protocol for Grain `Sampler` / `IndexSampler` instances."""


class GrainBatchOpProtocol(Protocol):
    """Opaque protocol for Grain `Batch` transformation instances."""


class GrainShardingCallableProtocol(Protocol):
    """Protocol for a Grain sharding class or factory."""

    def __call__(self) -> GrainShardingOptionsProtocol:
        """Instantiate sharding options."""
        ...


@runtime_checkable
class _GrainJaxProcessShardingProtocol(Protocol):
    """Protocol for `grain.python` exposing `sharding_by_jax_process`."""

    @property
    def sharding_by_jax_process(self) -> GrainShardingCallableProtocol | None:
        """Provide `sharding_by_jax_process` factory."""
        ...


@runtime_checkable
class _GrainLegacyDistShardingProtocol(Protocol):
    """Protocol for Grain doubles exposing `JAXDistributedSharding`."""

    @property
    def JAXDistributedSharding(self) -> GrainShardingCallableProtocol | None:
        """Provide the distributed sharding constructor."""
        ...


@runtime_checkable
class GrainModuleWithShardingProtocol(Protocol):
    """Protocol for a `grain.python` module exposing `NoSharding`."""

    @property
    def NoSharding(self) -> GrainShardingCallableProtocol | None:
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
class GrainModuleProtocol(
    GrainModuleWithShardingProtocol,
    GrainModuleClassesProtocol,
    Protocol,
):
    """Protocol for the `grain.python` module."""

    def IndexSampler(
        self,
        *,
        num_records: int,
        shard_options: GrainShardingOptionsProtocol | None,
        shuffle: bool = False,
        num_epochs: int = 1,
    ) -> GrainSamplerProtocol:
        """Create an `IndexSampler`."""
        ...

    def Batch(
        self,
        *,
        batch_size: int,
    ) -> GrainBatchOpProtocol:
        """Create a `Batch` operation.

        Args:
            batch_size: Target batch size.

        Returns:
            Constructed `GrainBatchOpProtocol`.

        """
        ...

    def DataLoader(
        self,
        *,
        data_source: HFDataSource,
        sampler: GrainSamplerProtocol,
        operations: Sequence[_DefaultBaseTransform | GrainBatchOpProtocol],
    ) -> (
        DataLoaderProtocol[Mapping[str, TensorType | Sequence[int]]]
        | Iterable[Mapping[str, TensorType | Sequence[int]]]
    ):
        """Construct a `DataLoader`."""
        ...


@runtime_checkable
class DuckDBDescribedCursorProtocol(Protocol):
    """Protocol for a DuckDB cursor or connection supporting `.fetchall()`."""

    def fetchall(self) -> list[tuple[JSONPrimitive, ...]]:
        """Fetch all result rows."""
        ...


@runtime_checkable
class DuckDBRawConnectionProtocol(Protocol):
    """Protocol for a raw DuckDB connection."""

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
    def Error(self) -> type[Exception] | str | None:
        """Provide the DuckDB base exception class."""
        ...

    def connect(self, database: str, *, read_only: bool = False) -> DuckDBRawConnectionProtocol:
        """Open a DuckDB database connection."""
        ...


@runtime_checkable
class DuckDBConnectCallableProtocol(Protocol):
    """Protocol for the `duckdb` module providing `.connect(...)`."""

    def connect(
        self,
        database: str = ":memory:",
        *,
        read_only: bool = False,
    ) -> DuckDBRawConnectionProtocol:
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

    def __getitem__(self, index: int) -> JSONDict | Mapping[str, JSONValue] | int | str:
        """Retrieve a single dataset element by index.

        Args:
            index: Zero-based element index.

        Returns:
            Dataset record at `index`.

        """
        return self._ds[index]


class _DefaultBaseTransform:
    """Base class for `BaseFormatTransform`, registered with `grain.python.MapTransform` when Grain is installed."""


try:
    from grain.python import MapTransform as _GrainMapTransformABC

    _GrainMapTransformABC.register(_DefaultBaseTransform)
except ImportError:
    pass


@runtime_checkable
class _SpecialTokenIdsProtocol(Protocol):
    """Protocol for tokenizers exposing `bos_token_id` and `eos_token_id`."""

    @property
    def bos_token_id(self) -> int | None:
        """Provide BOS token ID."""
        ...

    @property
    def eos_token_id(self) -> int | None:
        """Provide EOS token ID."""
        ...


class HFDataSource(_DefaultBaseSource):
    """Adapter turning a Hugging Face Dataset or sequence into a Grain `RandomAccessDataSource`."""


def _create_hf_data_source(base_ds: type = _DefaultBaseSource) -> type[HFDataSource]:
    """Return an `HFDataSource` class registered against `base_ds` when `base_ds` is an ABC.

    Args:
        base_ds: Optional Grain `RandomAccessDataSource` base class or ABC.

    Returns:
        `HFDataSource` class compatible with `base_ds`.

    """
    if isinstance(base_ds, abc.ABCMeta):
        base_ds.register(HFDataSource)
    return HFDataSource


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


def _resolve_bos_eos_ids(tokenizer: TokenizerProtocol) -> tuple[int, int]:
    """Extract integer `(bos_token_id, eos_token_id)` from `tokenizer` with `(2, 1)` fallback.

    Args:
        tokenizer: `TokenizerProtocol` instance.

    Returns:
        Tuple of `(bos_token_id, eos_token_id)`.

    """
    if isinstance(tokenizer, _SpecialTokenIdsProtocol):
        bos = tokenizer.bos_token_id if isinstance(tokenizer.bos_token_id, int) else 2
        eos = tokenizer.eos_token_id if isinstance(tokenizer.eos_token_id, int) else 1
        return (bos, eos)
    return (2, 1)


def _build_causal_token_dict(
    prompt_str: str,
    target_str: str,
    chosen_raw: JSONValue | ImageInput | AudioInput,
    rejected_raw: JSONValue | ImageInput | AudioInput,
    tokenizer: TokenizerProtocol,
) -> TokenizedElementDict:
    """Build aligned causal LM `input_ids`, `labels`, and `loss_mask` sequences from prompt and SQL strings.

    Concatenates `[BOS] + prompt_ids + sql_ids + [EOS]`, producing:
        - `input_ids` / `inputs`: `full_ids[:-1]`
        - `labels` / `targets`: `full_ids[1:]`
        - `loss_mask`: `0` on prompt token positions and `1` on SQL + `EOS` target positions.

    Args:
        prompt_str: Formatted prompt text.
        target_str: Target SQL query text.
        chosen_raw: Optional DPO chosen SQL completion.
        rejected_raw: Optional DPO rejected SQL completion.
        tokenizer: Active `TokenizerProtocol` instance.

    Returns:
        Aligned `TokenizedElementDict`.

    """
    bos_id, eos_id = _resolve_bos_eos_ids(tokenizer)
    prompt_ids = [int(t) for t in tokenizer.encode(prompt_str)]
    sql_ids = [int(t) for t in tokenizer.encode(target_str)]
    full_ids = [bos_id, *prompt_ids, *sql_ids, eos_id]
    input_ids = full_ids[:-1]
    labels = full_ids[1:]
    loss_mask = [0] * len(prompt_ids) + [1] * (len(sql_ids) + 1)

    res: TokenizedElementDict = {
        "input_ids": input_ids,
        "labels": labels,
        "loss_mask": loss_mask,
        "inputs": input_ids,
        "targets": labels,
    }
    if chosen_raw is not None or rejected_raw is not None:
        chosen_str = str(chosen_raw if chosen_raw is not None else target_str)
        rejected_str = str(rejected_raw if rejected_raw is not None else f"{target_str} -- rejected")
        chosen_ids = [int(t) for t in tokenizer.encode(chosen_str)]
        rejected_ids = [int(t) for t in tokenizer.encode(rejected_str)]
        res["chosen_inputs"] = [bos_id, *prompt_ids, *chosen_ids, eos_id]
        res["chosen_labels"] = [-100] * (1 + len(prompt_ids)) + [*chosen_ids, eos_id]
        res["rejected_inputs"] = [bos_id, *prompt_ids, *rejected_ids, eos_id]
        res["rejected_labels"] = [-100] * (1 + len(prompt_ids)) + [*rejected_ids, eos_id]
    return res


def tokenize_dataset_element(
    element: Mapping[str, JSONValue | ImageInput | AudioInput] | RawSQLDatasetRow,
    tokenizer: TokenizerProtocol,
    *,
    process_multimodal: bool = False,
    default_image_path: str | None = None,
    default_audio_path: str | None = None,
) -> TokenizedElementDict:
    """Extract and tokenize prompt, SQL target, preference pairs, and optional multimodal inputs from a dataset element.

    Args:
        element: Input dictionary element from dataset.
        tokenizer: `SQLTokenizer` or `TokenizerProtocol` instance.
        process_multimodal: Whether to format multimodal placeholders and extract image/audio tensors.
        default_image_path: Optional fallback image path from `ETLConfig.image_path`.
        default_audio_path: Optional fallback audio path from `ETLConfig.audio_path`.

    Returns:
        Dictionary containing aligned `'input_ids'`, `'labels'`, `'loss_mask'`, `'inputs'`, `'targets'`,
        optional preference pairs, and optional `'pixel_values'` / `'audio_values'`.

    """
    prompt = str(element.get("sql_prompt", element.get("question", element.get("prompt", ""))))
    target = str(element.get("sql", element.get("query", element.get("chosen", ""))))
    chosen_raw = element.get("chosen", element.get("chosen_sql"))
    rejected_raw = element.get("rejected", element.get("rejected_sql"))

    if not process_multimodal:
        return _build_causal_token_dict(prompt, target, chosen_raw, rejected_raw, tokenizer)

    from gemma_4_sql.backends.common_multimodal import (
        format_multimodal_prompt,
        process_audio,
        process_image,
    )

    image_input = element.get("image_bytes") or element.get("image_url") or element.get("image") or default_image_path
    audio_input = element.get("audio_clip") or element.get("audio") or default_audio_path

    formatted = format_multimodal_prompt(
        prompt,
        has_image=image_input is not None,
        has_audio=audio_input is not None,
    )
    resolved_prompt = str(formatted.get("prompt", prompt))
    res = _build_causal_token_dict(resolved_prompt, str(target), chosen_raw, rejected_raw, tokenizer)
    if image_input is not None and isinstance(image_input, (str, bytes, NDArrayProtocol)):
        img_data = process_image(image_input)
        res["pixel_values"] = img_data["pixel_values"]
    if audio_input is not None and isinstance(audio_input, (str, bytes, list, tuple, NDArrayProtocol)):
        aud_arg: str | bytes | Sequence[float] | NDArrayProtocol = (
            [float(x) for x in audio_input if isinstance(x, (int, float))]
            if isinstance(audio_input, (list, tuple))
            else audio_input
        )
        aud_data = process_audio(aud_arg)
        res["audio_values"] = aud_data["audio_values"]
    return res


class _BaseFormatTransformRoot(_DefaultBaseTransform):
    """Root class for format transforms with a typed `.map(element)` method."""

    tokenizer: TokenizerProtocol
    image_path: str | None
    audio_path: str | None

    def __init__(
        self,
        tokenizer: TokenizerProtocol,
        *,
        image_path: str | None = None,
        audio_path: str | None = None,
    ) -> None:
        """Initialize transform with tokenizer and optional default multimodal paths.

        Args:
            tokenizer: The SQL tokenizer instance.
            image_path: Optional fallback image path from `ETLConfig`.
            audio_path: Optional fallback audio path from `ETLConfig`.

        """
        self.tokenizer = tokenizer
        self.image_path = image_path
        self.audio_path = audio_path

    def map(
        self, element: Mapping[str, JSONValue | ImageInput | AudioInput] | RawSQLDatasetRow
    ) -> TokenizedElementDict:
        """Execute data mapping transformation with multimodal support.

        Args:
            element: Input dictionary element from dataset.

        Returns:
            Dictionary with tokenized inputs, targets, and optional multimodal features.

        """
        return tokenize_dataset_element(
            element,
            self.tokenizer,
            process_multimodal=True,
            default_image_path=self.image_path,
            default_audio_path=self.audio_path,
        )


class BaseFormatTransform(_BaseFormatTransformRoot):
    """Transforms dataset records into tokenized and multimodal array dictionaries."""


class MaxTextFormatTransform(BaseFormatTransform):
    """Transforms data into MaxText expected format with `segment_ids` and `positions` via composition."""

    def __init__(
        self,
        tokenizer: TokenizerProtocol,
        base_transform_cls: type[_BaseFormatTransformRoot] = BaseFormatTransform,
        *,
        image_path: str | None = None,
        audio_path: str | None = None,
    ) -> None:
        """Initialize MaxText transform with wrapped base transform.

        Args:
            tokenizer: The SQL tokenizer instance.
            base_transform_cls: Base transform class to delegate mapping to.
            image_path: Optional fallback image path from `ETLConfig`.
            audio_path: Optional fallback audio path from `ETLConfig`.

        """
        super().__init__(tokenizer, image_path=image_path, audio_path=audio_path)
        if image_path is not None or audio_path is not None:
            self._base: _BaseFormatTransformRoot = base_transform_cls(
                tokenizer, image_path=image_path, audio_path=audio_path
            )
        else:
            self._base = base_transform_cls(tokenizer)

    def map(
        self, element: Mapping[str, JSONValue | ImageInput | AudioInput] | RawSQLDatasetRow
    ) -> TokenizedElementDict:
        """Map an element with MaxText specific features (`segment_ids` and `positions`).

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


def wrap_maxtext_transform(
    base_transform_cls: type[_BaseFormatTransformRoot] = BaseFormatTransform,
) -> type[MaxTextFormatTransform]:
    """Return `MaxTextFormatTransform` configured to delegate to `base_transform_cls`.

    Args:
        base_transform_cls: Base transform class from `get_grain_classes`.

    Returns:
        `MaxTextFormatTransform` subclass delegating `.map(...)` through `base_transform_cls`.

    """
    if base_transform_cls is BaseFormatTransform:
        return MaxTextFormatTransform

    class _DelegatedMaxTextFormatTransform(MaxTextFormatTransform):
        """`MaxTextFormatTransform` bound to a custom `base_transform_cls`."""

        def __init__(
            self,
            tokenizer: TokenizerProtocol,
            *,
            image_path: str | None = None,
            audio_path: str | None = None,
        ) -> None:
            """Initialize with `base_transform_cls`."""
            super().__init__(
                tokenizer,
                base_transform_cls=base_transform_cls,
                image_path=image_path,
                audio_path=audio_path,
            )

    return _DelegatedMaxTextFormatTransform


def _create_base_format_transform(base_map: type = _DefaultBaseTransform) -> type[BaseFormatTransform]:
    """Return `BaseFormatTransform`, registering it against `base_map` if `base_map` is an ABC.

    Args:
        base_map: Optional Grain `MapTransform` base class or ABC.

    Returns:
        `BaseFormatTransform` class compatible with `base_map`.

    """
    if isinstance(base_map, abc.ABCMeta):
        base_map.register(BaseFormatTransform)
    return BaseFormatTransform


def get_grain_classes(
    grain_module: GrainModuleClassesProtocol | None,
) -> tuple[type[HFDataSource], type[BaseFormatTransform]]:
    """Provide Grain-compatible `(HFDataSource, BaseFormatTransform)` classes registered against `grain_module`.

    Args:
        grain_module: Optional loaded `grain.python` module.

    Returns:
        A tuple of `(HFDataSource, BaseFormatTransform)` classes.

    """
    if isinstance(grain_module, GrainModuleClassesProtocol):
        ds_cls = _create_hf_data_source(grain_module.RandomAccessDataSource)
        map_cls = _create_base_format_transform(grain_module.MapTransform)
        return (ds_cls, map_cls)
    return (HFDataSource, BaseFormatTransform)


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


def _make_grain_batch_op(
    grain_module: GrainModuleProtocol,
    batch_size: int,
) -> GrainBatchOpProtocol:
    """Construct a Grain `Batch` operation passing `batch_fn=collate_grain_batch` when accepted.

    Args:
        grain_module: Loaded `grain.python` module or double.
        batch_size: Target batch size.

    Returns:
        Constructed `Batch` operation.

    """
    if isinstance(grain_module, SubmoduleNamespaceProtocol):
        batch_ctor: Callable[..., GrainBatchOpProtocol] = vars(grain_module)["Batch"]
        return batch_ctor(batch_size=batch_size, batch_fn=collate_grain_batch)
    return grain_module.Batch(batch_size=batch_size)


def collate_grain_batch(
    elements: Sequence[TokenizedElementDict],
) -> dict[str, list[list[int]] | list[TokenizedElementValue]]:
    """Pad variable-length causal token sequences across a Grain batch into uniform 2D lists.

    Args:
        elements: Sequence of `TokenizedElementDict` records in the batch.

    Returns:
        Collated batch dictionary with zero-padded token sequences (`-100` for preference labels).

    """
    if not elements:
        return {}
    keys = list(elements[0].keys())
    collated: dict[str, list[list[int]] | list[TokenizedElementValue]] = {}
    for key in keys:
        vals = [el.get(key) for el in elements]
        if all(isinstance(v, list) and all(isinstance(x, int) for x in v) for v in vals):
            int_seqs: list[list[int]] = [[x for x in v if isinstance(x, int)] for v in vals if isinstance(v, list)]
            pad_tok = -100 if key in {"chosen_labels", "rejected_labels"} else 0
            max_len = max((len(seq) for seq in int_seqs), default=1)
            collated[key] = [seq + [pad_tok] * (max_len - len(seq)) for seq in int_seqs]
        else:
            collated[key] = vals
    return collated


def get_grain_sampler(
    source_len: int,
    distributed: bool,
    grain_module: GrainModuleProtocol | None,
) -> GrainSamplerProtocol:
    """Construct a Grain `IndexSampler` with optional JAX distributed sharding.

    Args:
        source_len: Number of records in the data source.
        distributed: Whether to enable distributed JAX process sharding.
        grain_module: Loaded `grain.python` module or `None`.

    Returns:
        Configured `grain.IndexSampler` instance.

    Raises:
        DependencyMissingError: If `grain_module` is `None`.

    """
    if grain_module is None:
        raise DependencyMissingError("Grain dependency is missing.", dependency_name="grain")
    shard_fn: GrainShardingCallableProtocol | None = None
    if distributed:
        if isinstance(grain_module, _GrainLegacyDistShardingProtocol) and callable(grain_module.JAXDistributedSharding):
            shard_fn = grain_module.JAXDistributedSharding
        elif isinstance(grain_module, _GrainJaxProcessShardingProtocol) and callable(
            grain_module.sharding_by_jax_process
        ):
            shard_fn = grain_module.sharding_by_jax_process
    else:
        shard_fn = grain_module.NoSharding
    shard_options: GrainShardingOptionsProtocol | None = shard_fn() if callable(shard_fn) else None
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
    grain_classes_fn: Callable[[GrainModuleProtocol | None], tuple[type[HFDataSource], type[BaseFormatTransform]]]
    | None = None,
    tokenizer_cls: Callable[..., TokenizerProtocol] = SQLTokenizer,
    sampler_fn: Callable[[int, bool], GrainSamplerProtocol] | None = None,
    transform_wrapper: Callable[[type[BaseFormatTransform]], type[BaseFormatTransform]] | None = None,
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
        """Load dataset using default Hugging Face or DuckDB loader.

        Args:
            ds: Dataset identifier or path.
            sp: Dataset split name.
            dp: Optional DuckDB file path.
            dt: Optional DuckDB table name.

        Returns:
            Loaded dataset sequence or list of record dictionaries.

        """
        return load_hf_or_duckdb_dataset(ds, sp, dp, dt, datasets_module)

    def _default_sampler(length: int, dist: bool) -> GrainSamplerProtocol:
        """Create default Grain sampler for dataset length and distributed setting.

        Args:
            length: Number of records in the dataset.
            dist: Whether distributed sharding is enabled.

        Returns:
            Configured `GrainSamplerProtocol` instance.

        """
        return get_grain_sampler(length, dist, grain_module)

    effective_load = load_fn or _default_load
    effective_classes: Callable[[GrainModuleProtocol | None], tuple[type[HFDataSource], type[BaseFormatTransform]]] = (
        grain_classes_fn if grain_classes_fn is not None else get_grain_classes
    )
    effective_sampler = sampler_fn or _default_sampler

    hf_dataset = effective_load(dataset_name, split, duckdb_path, duckdb_table)
    data_source_cls, transform_cls = effective_classes(grain_module)
    if transform_wrapper is not None:
        transform_cls = transform_wrapper(transform_cls)
    source = data_source_cls(hf_dataset)
    tokenizer = tokenizer_cls(model_name=tokenizer_name)
    sampler = effective_sampler(len(source), distributed)
    transform_inst = (
        transform_cls(tokenizer=tokenizer, image_path=config.image_path, audio_path=config.audio_path)
        if (config.image_path is not None or config.audio_path is not None)
        else transform_cls(tokenizer=tokenizer)
    )
    batch_op = _make_grain_batch_op(grain_module, batch_size)
    dataloader = grain_module.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=[transform_inst, batch_op],
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


@runtime_checkable
class _ColumnDescriptionCarrierProtocol(Protocol):
    """Protocol for a DB connection or cursor exposing `.description`."""

    @property
    def description(self) -> Sequence[tuple[str, ...]] | None:
        """Provide column metadata tuples."""
        ...


def _extract_column_descriptions(
    conn: DuckDBRawConnectionProtocol,
    cur: DuckDBDescribedCursorProtocol,
    default_cols: list[tuple[str, ...]],
) -> list[tuple[str, ...]]:
    """Extract column description tuples from `conn` or `cur`, falling back to `default_cols`.

    Args:
        conn: Raw DuckDB connection potentially carrying `.description`.
        cur: Executed DuckDB cursor potentially carrying `.description`.
        default_cols: Fallback column description tuples when neither carries metadata.

    Returns:
        List of column description tuples.

    """
    desc: Sequence[tuple[str, ...]] | None = (
        conn.description if isinstance(conn, _ColumnDescriptionCarrierProtocol) else None
    )
    if not isinstance(desc, (list, tuple)) or not desc:
        desc = cur.description if isinstance(cur, _ColumnDescriptionCarrierProtocol) else None
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
    from gemma_4_sql.exceptions import DatabaseExecutionError
    from gemma_4_sql.sdk.sql_lexer import validate_sql_identifier

    duckdb_module = _DUCKDB_LOADER.get_module()
    if duckdb_module is None or not isinstance(duckdb_module, DuckDBConnectCallableProtocol):
        msg = "duckdb is required. Install with `pip install duckdb`."
        raise DependencyMissingError(msg, dependency_name="duckdb")

    duckdb_error: type[Exception] = RuntimeError
    if isinstance(duckdb_module, DuckDBModuleWithErrorProtocol):
        raw_err = duckdb_module.Error
        if raw_err is not None and not isinstance(raw_err, str):
            duckdb_error = raw_err

    duckdb_errs: tuple[type[Exception], ...] = (
        duckdb_error,
        DatabaseExecutionError,
        ValueError,
        TypeError,
        OSError,
        RuntimeError,
        AttributeError,
    )
    raw_conn: DuckDBRawConnectionProtocol | None = None
    try:
        validated_table = validate_sql_identifier(table)
        raw_conn = duckdb_module.connect(db_path, read_only=True)
        cur = raw_conn.execute(f'SELECT * FROM "{validated_table}"')
        results: Sequence[Sequence[JSONPrimitive]] = cur.fetchall()
        default_cols: list[tuple[str, ...]] = [("col" + str(i),) for i in range(len(results[0]))] if results else []
        desc_list = _extract_column_descriptions(raw_conn, cur, default_cols)
        columns = [desc[0] for desc in desc_list]
        return [dict(zip(columns, row, strict=False)) for row in results]
    except Exception as e:
        if not isinstance(e, duckdb_errs):
            raise
        logger.exception("Failed to load dataset from DuckDB")
        msg = f"DuckDB error: {e}"
        raise ETLError(msg, cause=e) from e
    finally:
        if raw_conn is not None:
            raw_conn.close()


__all__ = [
    "BaseFormatTransform",
    "DatasetSequenceProtocol",
    "DuckDBConnectCallableProtocol",
    "DuckDBDescribedCursorProtocol",
    "DuckDBRawConnectionProtocol",
    "GrainModuleClassesProtocol",
    "GrainModuleProtocol",
    "HFDataSource",
    "HFDatasetsModuleProtocol",
    "MaxTextFormatTransform",
    "RawSQLDatasetRow",
    "TokenizedElementDict",
    "_BaseFormatTransformRoot",
    "_DefaultBaseSource",
    "_DefaultBaseTransform",
    "_create_base_format_transform",
    "_create_hf_data_source",
    "build_grain_dataloader",
    "get_grain_classes",
    "get_grain_sampler",
    "load_duckdb_dataset",
    "load_hf_or_duckdb_dataset",
    "pad_1d_list",
    "pad_token_batch",
    "resolve_duckdb_options",
    "tokenize_dataset_element",
    "wrap_maxtext_transform",
]

"""Unit and Grain integration tests for `gemma_4_sql.backends.common_data`."""

from __future__ import annotations

import abc
import importlib
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import overload
from unittest.mock import patch

import pytest

import gemma_4_sql.backends.common_data as common_data_mod
from gemma_4_sql.backends.common_data import (
    BaseFormatTransform,
    HFDataSource,
    MaxTextFormatTransform,
    RawSQLDatasetRow,
    TokenizedElementDict,
    _BaseFormatTransformRoot,
    _create_base_format_transform,
    _create_hf_data_source,
    build_grain_dataloader,
    get_grain_classes,
    get_grain_sampler,
    load_duckdb_dataset,
    load_hf_or_duckdb_dataset,
    pad_1d_list,
    pad_token_batch,
    resolve_duckdb_options,
    tokenize_dataset_element,
    wrap_maxtext_transform,
)
from gemma_4_sql.exceptions import DependencyMissingError, ETLError
from gemma_4_sql.tokenization import SQLTokenizer
from gemma_4_sql.type_hints import (
    AudioInput,
    ETLConfig,
    ImageInput,
    JSONDict,
    JSONPrimitive,
    JSONValue,
    MultimodalInput,
    TokenizerProtocol,
)


class _CustomRandomAccessABC(abc.ABC):
    """Custom ABC simulating a Grain `RandomAccessDataSource`."""


class _CustomMapTransformABC(abc.ABC):
    """Custom ABC simulating a Grain `MapTransform`."""


class _BareTokenizer:
    """Minimal `TokenizerProtocol` double without `bos_token_id` / `eos_token_id` attributes."""

    @overload
    def encode(self, text: str) -> list[int]:
        """Encode a single string."""

    @overload
    def encode(self, text: list[str] | tuple[str, ...]) -> list[list[int]]:
        """Encode a sequence of strings."""

    def encode(self, text: str | Sequence[str]) -> list[int] | list[list[int]]:
        """Encode string or sequence of strings into UTF-8 byte offsets.

        Returns:
            Offset token IDs for a single string or nested lists for a sequence.

        """
        if not isinstance(text, str):
            return [[b + 10 for b in item.encode("utf-8")] for item in text]
        return [b + 10 for b in text.encode("utf-8")]

    def decode(self, token_ids: Sequence[int], /) -> str:
        """Decode byte-offset token IDs back to UTF-8 string.

        Returns:
            Decoded UTF-8 text string.

        """
        return bytes(max(0, t - 10) for t in token_ids).decode("utf-8", errors="replace")


class _CustomGrainModule:
    """Typed Grain module double exposing `RandomAccessDataSource` and `MapTransform` ABCs."""

    RandomAccessDataSource = _CustomRandomAccessABC
    MapTransform = _CustomMapTransformABC


def test_real_grain_dataloader_and_causal_sequence_alignment() -> None:
    """Verify real `grain.DataLoader` iterates `HFDataSource` + `BaseFormatTransform` with aligned causal sequences."""
    import grain.python as grain

    tok = SQLTokenizer()
    assert _BareTokenizer().encode(["a"]) == [[107]]
    assert _BareTokenizer().decode([107]) == "a"

    data_source_cls, transform_cls = get_grain_classes(grain)
    assert issubclass(transform_cls, grain.MapTransform)
    assert issubclass(MaxTextFormatTransform, grain.MapTransform)

    records: list[JSONDict] = [
        {"sql_prompt": "SELECT users", "sql": "SELECT * FROM users;"},
        {"sql_prompt": "SELECT orders", "sql": "SELECT * FROM orders;"},
    ]
    source = data_source_cls(records)
    assert len(source) == 2
    assert source[0] == records[0]

    transform = transform_cls(tok)
    mapped0 = transform.map(records[0])
    prompt_ids = tok.encode("SELECT users")
    sql_ids = tok.encode("SELECT * FROM users;")
    expected_full = [tok.bos_token_id, *prompt_ids, *sql_ids, tok.eos_token_id]

    input_ids = mapped0["input_ids"]
    labels = mapped0["labels"]
    loss_mask = mapped0["loss_mask"]
    assert isinstance(input_ids, list)
    assert isinstance(labels, list)
    assert isinstance(loss_mask, list)
    assert input_ids == expected_full[:-1]
    assert labels == expected_full[1:]
    assert mapped0["inputs"] == expected_full[:-1]
    assert mapped0["targets"] == expected_full[1:]
    assert loss_mask == [0] * len(prompt_ids) + [1] * (len(sql_ids) + 1)
    assert len(input_ids) == len(labels) == len(loss_mask)

    class _RealGrainDatasets:
        """Dataset loader returning `records` for real `grain.DataLoader` iteration."""

        def load_dataset(self, path: str, *, split: str = "train") -> list[JSONDict]:
            """Return `records`."""
            _ = (path, split)
            return records

    from gemma_4_sql.backends.lazy_loader import optional_import

    raw_grain = optional_import("grain.python")
    assert isinstance(raw_grain, common_data_mod.GrainModuleProtocol)
    real_etl = build_grain_dataloader(
        config=ETLConfig(dataset_name="spider", split="train", batch_size=2),
        backend_name="jax",
        datasets_module=_RealGrainDatasets(),
        grain_module=raw_grain,
    )
    loader = real_etl.get("loader")
    assert loader is not None
    batches = list(loader)
    assert len(batches) == 1
    assert "input_ids" in batches[0]
    assert "labels" in batches[0]
    assert "loss_mask" in batches[0]

    ds_abc, map_abc = get_grain_classes(_CustomGrainModule())
    assert issubclass(ds_abc, _CustomRandomAccessABC)
    assert issubclass(map_abc, _CustomMapTransformABC)
    assert _create_hf_data_source(int) is HFDataSource
    assert _create_base_format_transform(int) is BaseFormatTransform
    ds_none, map_none = get_grain_classes(None)
    assert ds_none is HFDataSource
    assert map_none is BaseFormatTransform


def test_grain_import_error_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `common_data` reloads cleanly with fallback `_DefaultBaseTransform` when `grain` is absent."""
    grain_keys = [k for k in sys.modules if k == "grain" or k.startswith("grain.")]
    try:
        with monkeypatch.context() as mp_ctx:
            for k in grain_keys:
                mp_ctx.setitem(sys.modules, k, None)
            from typing import Protocol, runtime_checkable

            @runtime_checkable
            class _ReloadedCommonDataProtocol(Protocol):
                """Protocol for reloaded `common_data` module exposing `BaseFormatTransform`."""

                @property
                def BaseFormatTransform(self) -> type[common_data_mod.BaseFormatTransform]:
                    """Provide reloaded `BaseFormatTransform` class."""
                    ...

            reloaded = importlib.reload(common_data_mod)
            assert isinstance(reloaded, _ReloadedCommonDataProtocol)
            tok = SQLTokenizer()
            inst = reloaded.BaseFormatTransform(tok)
            out = inst.map({"sql_prompt": "q", "sql": "SELECT 1"})
            in_ids = out["input_ids"]
            lbls = out["labels"]
            mask = out["loss_mask"]
            assert isinstance(in_ids, Sequence) and not isinstance(in_ids, (str, bytes))
            assert isinstance(lbls, Sequence) and not isinstance(lbls, (str, bytes))
            assert isinstance(mask, Sequence) and not isinstance(mask, (str, bytes))
            assert len(in_ids) == len(lbls) == len(mask)
    finally:
        importlib.reload(common_data_mod)


def test_tokenize_dataset_element_bare_tokenizer_and_multimodal() -> None:
    """Verify `tokenize_dataset_element` handles bare `TokenizerProtocol`, preference pairs, and multimodal inputs."""
    bare_tok: TokenizerProtocol = _BareTokenizer()
    res_bare = tokenize_dataset_element(
        {"question": "count", "query": "SELECT 1", "chosen": "SELECT 1", "rejected": "SELECT 0"},
        bare_tok,
    )
    bare_in = res_bare["input_ids"]
    bare_lbl = res_bare["labels"]
    ch_in = res_bare["chosen_inputs"]
    ch_lbl = res_bare["chosen_labels"]
    rj_in = res_bare["rejected_inputs"]
    rj_lbl = res_bare["rejected_labels"]
    assert isinstance(bare_in, list) and isinstance(bare_lbl, list)
    assert isinstance(ch_in, list) and isinstance(ch_lbl, list)
    assert isinstance(rj_in, list) and isinstance(rj_lbl, list)
    assert bare_in[0] == 2
    assert bare_lbl[-1] == 1
    assert len(ch_in) == len(ch_lbl)
    assert len(rj_in) == len(rj_lbl)

    tok = SQLTokenizer()

    def _fake_fmt(prompt: str, *, has_image: bool = False, has_audio: bool = False) -> MultimodalInput:
        """Return formatted multimodal prompt."""
        prefix = ("<image> " if has_image else "") + ("<audio> " if has_audio else "")
        return {"prompt": f"{prefix}{prompt}", "modality": "multimodal"}

    with (
        patch("gemma_4_sql.backends.common_multimodal.format_multimodal_prompt", side_effect=_fake_fmt),
        patch(
            "gemma_4_sql.backends.common_multimodal.process_image",
            return_value={"pixel_values": [[[0.5]]], "patches": [[0.5]], "num_patches": 1, "shape": (3, 1, 1)},
        ),
        patch(
            "gemma_4_sql.backends.common_multimodal.process_audio",
            return_value={"audio_values": [0.25], "spectrogram": [[0.1]], "num_frames": 1, "sample_rate": 16000},
        ),
    ):
        res_mm = tokenize_dataset_element(
            {
                "sql_prompt": "q",
                "sql": "s",
                "chosen": "SELECT 1",
                "image_bytes": b"img",
                "audio_clip": [0.1, 0.2],
            },
            tok,
            process_multimodal=True,
        )
        assert res_mm["pixel_values"] == [[[0.5]]]
        assert res_mm["audio_values"] == [0.25]
        assert "chosen_inputs" in res_mm
        assert "rejected_inputs" in res_mm

        transform = BaseFormatTransform(tok)
        res_plain = transform.map({"question": "q2", "query": "s2"})
        assert "pixel_values" not in res_plain
        assert "audio_values" not in res_plain


def test_load_hf_or_duckdb_dataset_variants() -> None:
    """Verify `load_hf_or_duckdb_dataset` routes DuckDB vs Hugging Face datasets and checks missing dependencies."""
    calls: list[tuple[str, str]] = []

    def _duck_loader(path: str, table: str) -> list[JSONDict]:
        """Record DuckDB call and return single row.

        Returns:
            Single-element list of mock row dictionaries.

        """
        calls.append((path, table))
        return [{"id": 1}]

    res = load_hf_or_duckdb_dataset("ds", "train", "p.db", "tbl", None, load_duckdb_fn=_duck_loader)
    assert res == [{"id": 1}]
    assert calls == [("p.db", "tbl")]

    with pytest.raises(DependencyMissingError, match="Datasets dependency is missing"):
        load_hf_or_duckdb_dataset("ds", "train", None, None, None)

    class _FakeDatasetsModule:
        """Typed `datasets` module double."""

        def load_dataset(self, path: str, *, split: str = "train") -> list[JSONDict]:
            """Return deterministic single-row dataset."""
            return [{"dataset": path, "split": split}]

    res_hf = load_hf_or_duckdb_dataset("spider", "validation", None, None, _FakeDatasetsModule())
    assert res_hf == [{"dataset": "spider", "split": "validation"}]


class _FakeShardingOpts(common_data_mod.GrainShardingOptionsProtocol):
    """Typed Grain sharding options double."""

    def __init__(self, label: str) -> None:
        """Initialize with `label`."""
        self.label = label


class _FakeSamplerObj(common_data_mod.GrainSamplerProtocol):
    """Typed Grain sampler double."""

    def __init__(self, num_records: int, shard_options: common_data_mod.GrainShardingOptionsProtocol | None) -> None:
        """Store `num_records` and `shard_options`."""
        self.num_records = num_records
        self.shard_options = shard_options


class _FakeBatchOp(common_data_mod.GrainBatchOpProtocol):
    """Typed Grain batch operation double."""

    def __init__(self, batch_size: int) -> None:
        """Store `batch_size`."""
        self.batch_size = batch_size


class _FakeGrainModule:
    """Typed `GrainModuleProtocol` double for `get_grain_sampler` and `build_grain_dataloader`."""

    RandomAccessDataSource = _CustomRandomAccessABC
    MapTransform = _CustomMapTransformABC

    def __init__(self, *, non_callable_sharding: bool = False) -> None:
        """Initialize with optional `non_callable_sharding` toggle."""
        self.dist_sharding_calls = 0
        self.no_sharding_calls = 0
        self.last_batch_size = 0
        self._non_callable = non_callable_sharding

    @property
    def JAXDistributedSharding(self) -> Callable[[], _FakeShardingOpts] | None:
        """Provide distributed sharding factory or `None`."""
        if self._non_callable:
            return None

        def _make() -> _FakeShardingOpts:
            """Record distributed sharding call.

            Returns:
                `_FakeShardingOpts` configured for JAX distributed sharding.

            """
            self.dist_sharding_calls += 1
            return _FakeShardingOpts("jax_dist_sharding")

        return _make

    @property
    def NoSharding(self) -> Callable[[], _FakeShardingOpts] | None:
        """Provide `NoSharding` factory or `None`."""
        if self._non_callable:
            return None

        def _make() -> _FakeShardingOpts:
            """Record `NoSharding` call.

            Returns:
                `_FakeShardingOpts` configured for unsharded execution.

            """
            self.no_sharding_calls += 1
            return _FakeShardingOpts("no_sharding")

        return _make

    def IndexSampler(
        self,
        *,
        num_records: int,
        shard_options: common_data_mod.GrainShardingOptionsProtocol | None,
        shuffle: bool = False,
        num_epochs: int = 1,
    ) -> _FakeSamplerObj:
        """Return `_FakeSamplerObj`."""
        _ = (shuffle, num_epochs)
        return _FakeSamplerObj(num_records, shard_options)

    def Batch(self, *, batch_size: int) -> _FakeBatchOp:
        """Record `batch_size` and return `_FakeBatchOp`.

        Returns:
            `_FakeBatchOp` configured with `batch_size`.

        """
        self.last_batch_size = batch_size
        return _FakeBatchOp(batch_size)

    def DataLoader(
        self,
        *,
        data_source: HFDataSource,
        sampler: common_data_mod.GrainSamplerProtocol,
        operations: Sequence[common_data_mod._DefaultBaseTransform | common_data_mod.GrainBatchOpProtocol],
    ) -> list[dict[str, Sequence[int]]]:
        """Return single-batch list after exercising `data_source`."""
        _ = (sampler, operations)
        assert len(data_source) >= 0
        return [{"inputs": [1, 2], "targets": [2, 3]}]


def test_get_grain_sampler_and_build_grain_dataloader() -> None:
    """Verify `get_grain_sampler` and `build_grain_dataloader` across sharding, multimodal, and error branches."""
    with pytest.raises(DependencyMissingError, match="Grain dependency is missing"):
        get_grain_sampler(10, False, None)

    fake_grain = _FakeGrainModule()
    s_dist = get_grain_sampler(10, True, fake_grain)
    assert fake_grain.dist_sharding_calls == 1
    assert isinstance(s_dist, _FakeSamplerObj) and s_dist.num_records == 10

    s_local = get_grain_sampler(5, False, fake_grain)
    assert fake_grain.no_sharding_calls == 1
    assert isinstance(s_local, _FakeSamplerObj) and s_local.num_records == 5

    class _JaxProcessShardingGrain:
        """Grain double exposing `sharding_by_jax_process` without `JAXDistributedSharding`."""

        RandomAccessDataSource = _CustomRandomAccessABC
        MapTransform = _CustomMapTransformABC
        NoSharding = None

        @property
        def sharding_by_jax_process(self) -> Callable[[], _FakeShardingOpts] | None:
            """Factory producing `_FakeShardingOpts('jax_proc')`."""
            return lambda: _FakeShardingOpts("jax_proc")

        def IndexSampler(
            self,
            *,
            num_records: int,
            shard_options: common_data_mod.GrainShardingOptionsProtocol | None,
            shuffle: bool = False,
            num_epochs: int = 1,
        ) -> _FakeSamplerObj:
            """Return `_FakeSamplerObj`."""
            _ = (shuffle, num_epochs)
            return _FakeSamplerObj(num_records, shard_options)

        def Batch(self, *, batch_size: int) -> _FakeBatchOp:
            """Return `_FakeBatchOp`."""
            return _FakeBatchOp(batch_size)

        def DataLoader(
            self,
            *,
            data_source: HFDataSource,
            sampler: common_data_mod.GrainSamplerProtocol,
            operations: Sequence[common_data_mod._DefaultBaseTransform | common_data_mod.GrainBatchOpProtocol],
        ) -> list[dict[str, Sequence[int]]]:
            """Return empty batch list."""
            _ = (data_source, sampler, operations)
            return []

    jp_grain = _JaxProcessShardingGrain()
    assert jp_grain.Batch(batch_size=1).batch_size == 1
    assert jp_grain.DataLoader(data_source=HFDataSource([]), sampler=_FakeSamplerObj(0, None), operations=[]) == []
    s_real_dist = get_grain_sampler(4, True, jp_grain)
    assert isinstance(s_real_dist, _FakeSamplerObj) and s_real_dist.shard_options is not None

    class _NoDistShardingGrain(_JaxProcessShardingGrain):
        """Grain double with non-callable `sharding_by_jax_process`."""

        @property
        def sharding_by_jax_process(self) -> None:
            """Empty sharding property."""
            return

    s_none_proc = get_grain_sampler(4, True, _NoDistShardingGrain())
    assert isinstance(s_none_proc, _FakeSamplerObj) and s_none_proc.shard_options is None

    non_call_grain = _FakeGrainModule(non_callable_sharding=True)
    s_non_call = get_grain_sampler(5, True, non_call_grain)
    assert isinstance(s_non_call, _FakeSamplerObj) and s_non_call.shard_options is None
    s_non_call_local = get_grain_sampler(5, False, non_call_grain)
    assert isinstance(s_non_call_local, _FakeSamplerObj) and s_non_call_local.shard_options is None

    assert common_data_mod.collate_grain_batch([]) == {}
    collated_mm = common_data_mod.collate_grain_batch([
        {"inputs": [1, 2], "chosen_labels": [-100, 5], "pixel_values": [[[0.1]]]},
        {"inputs": [1], "chosen_labels": [-100], "pixel_values": [[[0.2]]]},
    ])
    assert collated_mm["inputs"] == [[1, 2], [1, 0]]
    assert collated_mm["chosen_labels"] == [[-100, 5], [-100, -100]]
    assert collated_mm["pixel_values"] == [[[[0.1]]], [[[0.2]]]]

    with pytest.raises(DependencyMissingError, match="Missing grain or datasets"):
        build_grain_dataloader(
            config=ETLConfig(dataset_name="ds", split="train", batch_size=2),
            backend_name="jax",
            datasets_module=None,
            grain_module=fake_grain,
        )

    class _FakeDatasets:
        """Dataset loader double returning deterministic SQL rows."""

        def load_dataset(self, path: str, *, split: str = "train") -> list[JSONDict]:
            """Return single sample row."""
            _ = path
            return [{"sql_prompt": f"q_{split}", "sql": "SELECT 1"}]

    res_ok = build_grain_dataloader(
        config=ETLConfig(dataset_name="ds", split="train", batch_size=4),
        backend_name="jax",
        datasets_module=_FakeDatasets(),
        grain_module=fake_grain,
    )
    assert res_ok["status"] == "loaded"
    assert fake_grain.last_batch_size == 4

    res_mm_mt = build_grain_dataloader(
        config=ETLConfig(
            dataset_name="ds",
            split="train",
            batch_size=2,
            distributed=True,
            image_path="schema.png",
        ),
        backend_name="maxtext",
        datasets_module=_FakeDatasets(),
        grain_module=fake_grain,
        transform_wrapper=wrap_maxtext_transform,
        extra_options={"stage": "sft"},
    )
    assert res_mm_mt["status"] == "loaded"
    assert res_mm_mt.get("distributed") is True


def test_pad_batch_duckdb_options_and_maxtext_transform(tmp_path: Path) -> None:
    """Verify batch padding, DuckDB options, MaxText transform wrapping, and DuckDB dataset loading.

    Exercises `pad_1d_list`, `pad_token_batch`, `resolve_duckdb_options`, `wrap_maxtext_transform`,
    and `load_duckdb_dataset`.
    """
    assert pad_1d_list([]) == []
    assert pad_1d_list([[1, 2], [3]]) == [[1, 2], [3, 0]]
    padded = pad_token_batch([[1, 2], [3]], [[4], [5, 6, 7]])
    assert padded == {"inputs": [[1, 2], [3, 0]], "targets": [[4, 0, 0], [5, 6, 7]]}

    cfg = ETLConfig(dataset_name="d", split="train")
    assert resolve_duckdb_options(cfg, {}) == (None, None)
    assert resolve_duckdb_options(cfg, {"duckdb_path": "a.db", "duckdb_table": "t"}) == ("a.db", "t")

    tok = SQLTokenizer()
    mt_cls = wrap_maxtext_transform(BaseFormatTransform)
    mt_inst = mt_cls(tok)
    mt_mapped = mt_inst.map({"sql_prompt": "q", "sql": "SELECT 1"})
    mt_inputs = mt_mapped["inputs"]
    assert isinstance(mt_inputs, list)
    expected_len = len(mt_inputs)
    assert mt_mapped["segment_ids"] == [1] * expected_len
    assert mt_mapped["positions"] == list(range(expected_len))

    class _CustomTransformSub(BaseFormatTransform):
        """Custom `BaseFormatTransform` subclass to exercise `_DelegatedMaxTextFormatTransform`."""

        def __init__(
            self,
            tokenizer: TokenizerProtocol,
            *,
            image_path: str | None = None,
            audio_path: str | None = None,
        ) -> None:
            """Store tokenizer."""
            super().__init__(tokenizer, image_path=image_path, audio_path=audio_path)

        def map(
            self, element: Mapping[str, JSONValue | ImageInput | AudioInput] | RawSQLDatasetRow
        ) -> TokenizedElementDict:
            """Return fixed 2-token mapping."""
            _ = element
            return {"inputs": [10, 20], "targets": [20, 1]}

    delegated_cls = wrap_maxtext_transform(_CustomTransformSub)
    delegated_inst = delegated_cls(tok, image_path="img.png")
    delegated_mapped = delegated_inst.map({"sql_prompt": "q", "sql": "SELECT 1"})
    assert delegated_mapped["segment_ids"] == [1, 1]
    assert delegated_mapped["positions"] == [0, 1]

    class _MinimalTransformSub(_BaseFormatTransformRoot):
        """1-arg `_BaseFormatTransformRoot` subclass without `image_path` / `audio_path` kwargs."""

        def __init__(self, tokenizer: TokenizerProtocol) -> None:
            """Store tokenizer only."""
            self.tokenizer = tokenizer

        def map(
            self, element: Mapping[str, JSONValue | ImageInput | AudioInput] | RawSQLDatasetRow
        ) -> TokenizedElementDict:
            """Return non-list inputs to cover `not isinstance(inputs, list)` branch."""
            _ = element
            return {"inputs": 1, "targets": [1]}

    min_mt = MaxTextFormatTransform(tok, base_transform_cls=_MinimalTransformSub)
    min_mapped = min_mt.map({"sql_prompt": "q", "sql": "SELECT 1"})
    assert min_mapped["segment_ids"] == [1]
    assert min_mapped["positions"] == [0]

    import duckdb

    real_db = tmp_path / "etl_real.duckdb"
    with duckdb.connect(str(real_db)) as setup_conn:
        setup_conn.execute("CREATE TABLE sql_pairs (sql_prompt VARCHAR, sql VARCHAR)")
        setup_conn.execute("INSERT INTO sql_pairs VALUES ('list users', 'SELECT * FROM users')")
    real_rows = load_duckdb_dataset(str(real_db), "sql_pairs")
    assert real_rows == [{"sql_prompt": "list users", "sql": "SELECT * FROM users"}]

    with pytest.raises(ETLError):
        load_duckdb_dataset(":memory:", "nonexistent_tbl")

    class _CursorOnlyDesc:
        """Cursor exposing `.description` and `.fetchall()` when connection lacks `.description`."""

        @property
        def description(self) -> list[tuple[str]]:
            """Provide cursor column descriptions."""
            return [("sql_prompt",), ("sql",)]

        def fetchall(self) -> list[tuple[JSONPrimitive, ...]]:
            """Return single sample row."""
            return [("count users", "SELECT COUNT(*) FROM users")]

    class _CursorNoDesc:
        """Cursor without `.description` triggering `default_cols` fallback."""

        def fetchall(self) -> list[tuple[JSONPrimitive, ...]]:
            """Return single 2-column tuple."""
            return [("q", "SELECT 1")]

    class _ConnWithoutDesc:
        """Connection without `.description` returning `_CursorOnlyDesc` or `_CursorNoDesc`."""

        def __init__(self, *, empty_desc: bool = False) -> None:
            """Initialize with optional empty cursor description."""
            self._empty_desc = empty_desc
            self.closed = False

        def execute(self, _query: str) -> _CursorOnlyDesc | _CursorNoDesc:
            """Return cursor double."""
            return _CursorNoDesc() if self._empty_desc else _CursorOnlyDesc()

        def close(self) -> None:
            """Mark connection closed."""
            self.closed = True

    from tests.utils.test_utils import FakeModuleNamespace

    class _FakeDuckDBModule(FakeModuleNamespace):
        """Typed `duckdb` module double with `.Error` and `.connect`."""

        Error: type[Exception] | str | None = duckdb.Error

        def __init__(self, conn: _ConnWithoutDesc, *, non_exc_error: bool = False) -> None:
            """Store connection double."""
            super().__init__("duckdb")
            self._conn = conn
            if non_exc_error:
                self.Error = "not_an_exception"

        def connect(self, database: str = ":memory:", *, read_only: bool = False) -> _ConnWithoutDesc:
            """Return stored connection double."""
            _ = (database, read_only)
            return self._conn

    from gemma_4_sql.backends.lazy_loader import override_lazy_module

    conn1 = _ConnWithoutDesc(empty_desc=False)
    with override_lazy_module("duckdb", _FakeDuckDBModule(conn1)):
        rows1 = load_duckdb_dataset("fake.db", "valid_tbl")
        assert rows1 == [{"sql_prompt": "count users", "sql": "SELECT COUNT(*) FROM users"}]
        assert conn1.closed is True

    conn2 = _ConnWithoutDesc(empty_desc=True)
    with override_lazy_module("duckdb", _FakeDuckDBModule(conn2, non_exc_error=True)):
        rows2 = load_duckdb_dataset("fake.db", "valid_tbl")
        assert rows2 == [{"col0": "q", "col1": "SELECT 1"}]

    class _BareDuckDBModule(FakeModuleNamespace):
        """DuckDB module double without `.Error` attribute."""

        def __init__(self, conn: _ConnWithoutDesc) -> None:
            """Store connection double."""
            super().__init__("duckdb")
            self._conn = conn

        def connect(self, database: str = ":memory:", *, read_only: bool = False) -> _ConnWithoutDesc:
            """Return stored connection double."""
            _ = (database, read_only)
            return self._conn

    conn3 = _ConnWithoutDesc(empty_desc=False)
    with override_lazy_module("duckdb", _BareDuckDBModule(conn3)):
        rows3 = load_duckdb_dataset("fake.db", "valid_tbl")
        assert len(rows3) == 1

    with override_lazy_module("duckdb", None), pytest.raises(DependencyMissingError, match="duckdb is required"):
        load_duckdb_dataset("fake.db", "valid_tbl")

    class _KeyErrorDuckDBModule(FakeModuleNamespace):
        """DuckDB module double raising non-operational `KeyError` on `connect`."""

        def connect(self, database: str = ":memory:", *, read_only: bool = False) -> _ConnWithoutDesc:
            """Raise `KeyError` to verify programming bugs propagate uncaught.

            Raises:
                KeyError: Always raised.

            """
            _ = (database, read_only)
            raise KeyError("unexpected_duckdb_bug")

    with (
        override_lazy_module("duckdb", _KeyErrorDuckDBModule("duckdb")),
        pytest.raises(KeyError, match="unexpected_duckdb_bug"),
    ):
        load_duckdb_dataset("fake.db", "valid_tbl")

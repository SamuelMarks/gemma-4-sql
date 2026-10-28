"""Tests for common_data.py."""

from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.backends.common_data import (
    _create_base_format_transform,
    _create_hf_data_source,
    get_grain_classes,
    load_duckdb_dataset,
)


def test_create_hf_data_source() -> None:
    """Test _create_hf_data_source."""

    class DummyBase:
        """Test stub for `DummyBase`."""

    HFDataSource = _create_hf_data_source(DummyBase)

    hf_ds_mock = [1, 2, 3]
    ds = HFDataSource(hf_ds_mock)

    assert len(ds) == 3
    assert ds[0] == 1
    assert ds[1] == 2
    assert ds[2] == 3


def test_create_base_format_transform() -> None:
    """Test _create_base_format_transform."""

    class DummyBaseMap:
        """Test stub for `DummyBaseMap`."""

    BaseFormatTransform = _create_base_format_transform(DummyBaseMap)

    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.side_effect = lambda x: [len(x)]  # pyright: ignore[reportUnknownLambdaType]

    transform = BaseFormatTransform(mock_tokenizer)

    # Text only
    element_text = {"sql_prompt": "test prompt", "sql": "SELECT * FROM test"}
    res_text = transform.map(element_text)
    assert "inputs" in res_text
    assert "targets" in res_text
    assert "pixel_values" not in res_text
    assert "audio_values" not in res_text

    # With image and audio mock
    element_multi = {"question": "multimodal question", "query": "SELECT image", "image": "dummy_image_data", "audio": "dummy_audio_data"}

    with patch("gemma_4_sql.backends.common_multimodal.format_multimodal_prompt") as mock_fmt, patch("gemma_4_sql.backends.common_multimodal.process_image") as mock_pi, patch("gemma_4_sql.backends.common_multimodal.process_audio") as mock_pa:
        mock_fmt.return_value = {"prompt": "formatted multimodal question"}
        mock_pi.return_value = {"pixel_values": "pixels"}
        mock_pa.return_value = {"audio_values": "audio"}

        res_multi = transform.map(element_multi)

        assert "pixel_values" in res_multi
        assert res_multi["pixel_values"] == "pixels"
        assert "audio_values" in res_multi
        assert res_multi["audio_values"] == "audio"


def test_get_grain_classes() -> None:
    """Test get_grain_classes."""

    class CustomDS:
        """Custom data source base class."""

    class CustomMap:
        """Custom map transform base class."""

    mock_module = MagicMock()
    mock_module.RandomAccessDataSource = CustomDS
    mock_module.MapTransform = CustomMap

    DataSource, MapTransform = get_grain_classes(mock_module)
    assert issubclass(DataSource, CustomDS)
    assert issubclass(MapTransform, CustomMap)

    # Test without attributes
    mock_module2 = MagicMock(spec=[])
    DataSource2, MapTransform2 = get_grain_classes(mock_module2)
    assert isinstance(DataSource2, type)
    assert isinstance(MapTransform2, type)


class _MockDuckDBModule:
    """Typed mock DuckDB module satisfying DuckDBConnectCallableProtocol."""

    Error: type[Exception] = RuntimeError

    def __init__(self, conn: MagicMock | None = None, side_effect: Exception | None = None) -> None:
        """Initialize with optional mock connection or side_effect."""
        self.conn = conn or MagicMock()
        self.side_effect = side_effect
        self.calls: list[tuple[str, bool]] = []

    def connect(self, database: str, *, read_only: bool = False) -> MagicMock:
        """Record connect call and return mock connection or raise side_effect."""
        self.calls.append((database, read_only))
        if self.side_effect is not None:
            raise self.side_effect
        return self.conn


def test_load_duckdb_dataset_success() -> None:
    """Test load_duckdb_dataset success."""
    mock_conn = MagicMock()
    mock_duckdb = _MockDuckDBModule(conn=mock_conn)
    mock_conn.execute.return_value.fetchall.return_value = [(1, "test")]
    mock_conn.description = [("id",), ("name",)]

    with patch("gemma_4_sql.backends.lazy_loader.LazyLoader.get_module", return_value=mock_duckdb):
        res = load_duckdb_dataset("test.db", "test_table")
        assert res == [{"id": 1, "name": "test"}]
        assert mock_duckdb.calls == [("test.db", True)]
        mock_conn.execute.assert_called_once_with('SELECT * FROM "test_table"')
        mock_conn.close.assert_called_once()


def test_load_duckdb_dataset_fallback_description() -> None:
    """Test load_duckdb_dataset fallback description."""
    mock_conn = MagicMock()
    del mock_conn.description
    mock_duckdb = _MockDuckDBModule(conn=mock_conn)
    mock_conn.execute.return_value.fetchall.return_value = [(1, "test")]

    with patch("gemma_4_sql.backends.lazy_loader.LazyLoader.get_module", return_value=mock_duckdb):
        res = load_duckdb_dataset("test.db", "test_table")
        assert res == [{"col0": 1, "col1": "test"}]


def test_load_duckdb_dataset_missing_module() -> None:
    """Test load_duckdb_dataset missing module."""
    with patch("gemma_4_sql.backends.lazy_loader.LazyLoader.get_module", return_value=None), pytest.raises(RuntimeError, match="duckdb is required"):
        load_duckdb_dataset("test.db", "test_table")


def test_load_duckdb_dataset_invalid_table() -> None:
    """Test load_duckdb_dataset invalid table."""
    mock_duckdb = _MockDuckDBModule()
    with patch("gemma_4_sql.backends.lazy_loader.LazyLoader.get_module", return_value=mock_duckdb), pytest.raises(RuntimeError, match="DuckDB error: Invalid or unsafe table name:"):
        load_duckdb_dataset("test.db", "invalid table")


def test_load_duckdb_dataset_exception() -> None:
    """Test load_duckdb_dataset exception."""
    mock_duckdb = _MockDuckDBModule(side_effect=RuntimeError("duckdb error"))
    with patch("gemma_4_sql.backends.lazy_loader.LazyLoader.get_module", return_value=mock_duckdb), pytest.raises(RuntimeError, match="DuckDB error"):
        load_duckdb_dataset("test.db", "test_table")


def test_load_hf_or_duckdb_and_grain_dataloader() -> None:
    """Verify load_hf_or_duckdb_dataset, get_grain_sampler, and build_grain_dataloader."""
    from gemma_4_sql.backends.common_data import (
        build_grain_dataloader,
        get_grain_sampler,
        load_hf_or_duckdb_dataset,
    )
    from gemma_4_sql.exceptions import DependencyMissingError
    from gemma_4_sql.type_hints import ETLConfig

    assert load_hf_or_duckdb_dataset("ds", "train", "db.duckdb", "tbl", None, load_duckdb_fn=lambda p, t: [{"db": f"{p}:{t}"}]) == [{"db": "db.duckdb:tbl"}]
    with pytest.raises(DependencyMissingError):
        load_hf_or_duckdb_dataset("ds", "train", None, None, None)

    mock_datasets = MagicMock()
    mock_datasets.load_dataset.return_value = [{"sql_prompt": "q", "sql": "SELECT 1"}]
    loaded_ds = load_hf_or_duckdb_dataset("ds", "train", None, None, mock_datasets)
    assert len(loaded_ds) == 1

    with pytest.raises(DependencyMissingError):
        get_grain_sampler(10, False, None)

    mock_grain = MagicMock()
    assert get_grain_sampler(10, True, mock_grain) is not None
    assert get_grain_sampler(10, False, mock_grain) is not None

    cfg = ETLConfig(dataset_name="ds", split="train", batch_size=2, distributed=False)
    with pytest.raises(DependencyMissingError):
        build_grain_dataloader(cfg, "jax", None, mock_grain)

    res = build_grain_dataloader(cfg, "jax", mock_datasets, mock_grain, transform_wrapper=lambda cls: cls)
    assert res["status"] == "loaded"
    assert res["backend"] == "jax"

    res_no_wrap = build_grain_dataloader(cfg, "jax", mock_datasets, mock_grain)
    assert res_no_wrap["status"] == "loaded"

    from gemma_4_sql.backends.common_data import tokenize_dataset_element, wrap_maxtext_transform
    from gemma_4_sql.tokenization import SQLTokenizer

    tok = SQLTokenizer()
    tok_res = tokenize_dataset_element({"sql_prompt": "q", "sql": "SELECT 1"}, tok, process_multimodal=False)
    assert "inputs" in tok_res and "targets" in tok_res

    pref_res_nomm = tokenize_dataset_element(
        {"sql_prompt": "q", "chosen": "SELECT 1", "rejected": "SELECT 2"},
        tok,
        process_multimodal=False,
    )
    assert "chosen_inputs" in pref_res_nomm and "rejected_inputs" in pref_res_nomm
    assert pref_res_nomm["chosen_inputs"] != pref_res_nomm["rejected_inputs"]

    pref_res_mm = tokenize_dataset_element(
        {"sql_prompt": "q", "chosen": "SELECT 1", "rejected": "SELECT 2"},
        tok,
        process_multimodal=True,
    )
    assert "chosen_inputs" in pref_res_mm and "rejected_inputs" in pref_res_mm
    assert pref_res_mm["chosen_inputs"] != pref_res_mm["rejected_inputs"]

    _, base_transform_cls = get_grain_classes(None)
    mt_cls = wrap_maxtext_transform(base_transform_cls)
    mt_inst = mt_cls(tok)
    mt_mapped = mt_inst.map({"sql_prompt": "q", "sql": "SELECT 1"})
    assert mt_mapped["segment_ids"] == [1]
    assert mt_mapped["positions"] == [0]

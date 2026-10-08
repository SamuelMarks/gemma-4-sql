"""Tests for common_data.py"""

from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.backends.common_data import (
    _create_base_format_transform,
    _create_hf_data_source,
    get_grain_classes,
    load_duckdb_dataset,
)


def test_create_hf_data_source():
    """Test _create_hf_data_source."""

    class DummyBase:
        """Docstring for DummyBase."""

    HFDataSource = _create_hf_data_source(DummyBase)

    hf_ds_mock = [1, 2, 3]
    ds = HFDataSource(hf_ds_mock)

    assert len(ds) == 3
    assert ds[0] == 1
    assert ds[1] == 2
    assert ds[2] == 3


def test_create_base_format_transform():
    """Test _create_base_format_transform."""

    class DummyBaseMap:
        """Docstring for DummyBaseMap."""

    BaseFormatTransform = _create_base_format_transform(DummyBaseMap)

    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.side_effect = lambda x: [len(x)]

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


def test_get_grain_classes():
    """Test get_grain_classes."""
    mock_module = MagicMock()
    mock_module.RandomAccessDataSource = int
    mock_module.MapTransform = float

    DataSource, MapTransform = get_grain_classes(mock_module)
    assert issubclass(DataSource, int)
    assert issubclass(MapTransform, float)

    # Test without attributes
    mock_module2 = MagicMock(spec=[])
    DataSource2, MapTransform2 = get_grain_classes(mock_module2)
    assert issubclass(DataSource2, object)
    assert issubclass(MapTransform2, object)


def test_load_duckdb_dataset_success():
    """Test load_duckdb_dataset success."""
    mock_duckdb = MagicMock()
    mock_conn = MagicMock()
    mock_duckdb.connect.return_value = mock_conn
    mock_conn.execute.return_value.fetchall.return_value = [(1, "test")]
    mock_conn.description = [("id",), ("name",)]

    with patch("gemma_4_sql.backends.lazy_loader.LazyLoader.get_module", return_value=mock_duckdb):
        res = load_duckdb_dataset("test.db", "test_table")
        assert res == [{"id": 1, "name": "test"}]
        mock_duckdb.connect.assert_called_once_with("test.db", read_only=True)
        mock_conn.execute.assert_called_once_with('SELECT * FROM "test_table"')
        mock_conn.close.assert_called_once()


def test_load_duckdb_dataset_fallback_description():
    """Test load_duckdb_dataset fallback description."""
    mock_duckdb = MagicMock()
    mock_conn = MagicMock()
    del mock_conn.description
    mock_duckdb.connect.return_value = mock_conn
    mock_conn.execute.return_value.fetchall.return_value = [(1, "test")]

    with patch("gemma_4_sql.backends.lazy_loader.LazyLoader.get_module", return_value=mock_duckdb):
        res = load_duckdb_dataset("test.db", "test_table")
        assert res == [{"col0": 1, "col1": "test"}]


def test_load_duckdb_dataset_missing_module():
    """Test load_duckdb_dataset missing module."""
    with patch("gemma_4_sql.backends.lazy_loader.LazyLoader.get_module", return_value=None), pytest.raises(RuntimeError, match="duckdb is required"):
        load_duckdb_dataset("test.db", "test_table")


def test_load_duckdb_dataset_invalid_table():
    """Test load_duckdb_dataset invalid table."""
    mock_duckdb = MagicMock()
    with patch("gemma_4_sql.backends.lazy_loader.LazyLoader.get_module", return_value=mock_duckdb), pytest.raises(RuntimeError, match="DuckDB error: Invalid or unsafe table name:"):
        load_duckdb_dataset("test.db", "invalid table")


def test_load_duckdb_dataset_exception():
    """Test load_duckdb_dataset exception."""
    mock_duckdb = MagicMock()
    mock_duckdb.connect.side_effect = Exception("duckdb error")
    with patch("gemma_4_sql.backends.lazy_loader.LazyLoader.get_module", return_value=mock_duckdb), pytest.raises(RuntimeError, match="DuckDB error"):
        load_duckdb_dataset("test.db", "test_table")

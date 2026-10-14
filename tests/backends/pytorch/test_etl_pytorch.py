"""Tests for PyTorch ETL pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import requires_torch

pytestmark = requires_torch

from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import ETLConfig


def test_pytorch_etl_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `pytorch.etl` handles missing `datasets`, `torch`, `DataLoader`, and `Dataset` attributes."""
    import gemma_4_sql.backends.pytorch.etl as etl_module

    monkeypatch.setattr(etl_module, "datasets", None)
    monkeypatch.setattr(etl_module, "torch", None)
    monkeypatch.setattr(etl_module, "DataLoader", None)
    monkeypatch.setattr(etl_module, "Dataset", None)
    assert etl_module.datasets is None
    assert etl_module.torch is None
    assert etl_module.DataLoader is None
    assert etl_module.Dataset is None


def test_pytorch_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `PyTorchDataset` tokenizes SQL records and extracts multimodal image and audio features."""
    import gemma_4_sql.backends.pytorch.etl as etl_module

    mock_torch = MagicMock()
    mock_torch.long = "long"
    mock_torch.float32 = "float32"
    mock_torch.tensor.side_effect = lambda x, dtype: (x, dtype)
    monkeypatch.setattr(etl_module, "torch", mock_torch)

    mock_ds = [{"sql_prompt": "p1", "sql": "s1"}, {"question": "p2", "query": "s2", "image_url": "url", "audio_clip": "aud"}]

    mock_tok = MagicMock()
    mock_tok.encode.side_effect = lambda x: [x]

    cls = etl_module._get_pytorch_classes()
    dataset = cls(mock_ds, mock_tok)

    assert len(dataset) == 2

    with patch("gemma_4_sql.backends.common_multimodal.format_multimodal_prompt") as mock_format:
        mock_format.return_value = {"prompt": "formatted_p1"}
        item1 = dataset[0]
        assert item1["inputs"] == (["formatted_p1"], "long")
        assert item1["targets"] == (["s1"], "long")

        with patch("gemma_4_sql.backends.common_multimodal.process_image") as mock_img:
            mock_img.return_value = {"pixel_values": "pixels"}
            with patch("gemma_4_sql.backends.common_multimodal.process_audio") as mock_aud:
                mock_aud.return_value = {"audio_values": "audio"}

                mock_format.return_value = {}
                item2 = dataset[1]
                assert item2["inputs"] == (["p2"], "long")
                assert item2["targets"] == (["s2"], "long")
                assert item2["pixel_values"] == ("pixels", "float32")
                assert item2["audio_values"] == ("audio", "float32")

                mock_img.assert_called_with("url")
                mock_aud.assert_called_with("aud")


def test_collate_fn(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `_collate_fn` pads sequences and stacks optional multimodal tensors."""
    import gemma_4_sql.backends.pytorch.etl as etl_module

    mock_torch = MagicMock()
    monkeypatch.setattr(etl_module, "torch", mock_torch)
    mock_torch.stack.side_effect = lambda x: x
    mock_torch.nn.utils.rnn.pad_sequence.side_effect = lambda x, batch_first: x

    batch = [{"inputs": 1, "targets": 2, "pixel_values": 3, "audio_values": 4}, {"inputs": 5, "targets": 6, "pixel_values": 7, "audio_values": 8}]

    res = etl_module._collate_fn(batch)
    assert res["inputs"] == [1, 5]
    assert res["targets"] == [2, 6]
    assert res["pixel_values"] == [3, 7]
    assert res["audio_values"] == [4, 8]

    batch2 = [{"inputs": 1, "targets": 2}, {"inputs": 5, "targets": 6}]
    res2 = etl_module._collate_fn(batch2)
    assert res2["inputs"] == [1, 5]
    assert "pixel_values" not in res2


def test_get_sampler() -> None:
    """Verify `_get_sampler` returns `DistributedSampler` when distributed is enabled and `None` otherwise."""
    import gemma_4_sql.backends.pytorch.etl as etl_module

    assert etl_module._get_sampler("ds", False) is None

    with patch("builtins.__import__") as mock_import:
        mock_dist = MagicMock()
        mock_dist.DistributedSampler.return_value = "sampler"
        mock_import.return_value = mock_dist

        assert etl_module._get_sampler("ds", True) == "sampler"

        mock_dist.DistributedSampler.side_effect = RuntimeError("error")
        assert etl_module._get_sampler("ds", True) is None


def test_load_hf_or_duckdb(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `_load_hf_or_duckdb` loads from DuckDB or Hugging Face `datasets`."""
    import gemma_4_sql.backends.pytorch.etl as etl_module

    with patch("gemma_4_sql.backends.pytorch.etl.load_duckdb_dataset") as mock_load:
        mock_load.return_value = "duck"
        assert etl_module._load_hf_or_duckdb("ds", "split", "path", "table") == "duck"

    mock_datasets = MagicMock()
    monkeypatch.setattr(etl_module, "datasets", mock_datasets)
    mock_datasets.load_dataset.return_value = "hf"

    assert etl_module._load_hf_or_duckdb("ds", "split", None, None) == "hf"

    monkeypatch.setattr(etl_module, "datasets", None)
    with pytest.raises(DependencyMissingError):
        etl_module._load_hf_or_duckdb("ds", "split", None, None)


def test_build_dataloader(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `build_dataloader` constructs a PyTorch DataLoader and raises `DependencyMissingError` when missing."""
    import gemma_4_sql.backends.pytorch.etl as etl_module

    monkeypatch.setattr(etl_module, "datasets", MagicMock())
    monkeypatch.setattr(etl_module, "torch", MagicMock())
    monkeypatch.setattr(etl_module, "Dataset", MagicMock())
    monkeypatch.setattr(etl_module, "DataLoader", MagicMock())

    config = ETLConfig(dataset_name="ds", split="train")

    with (
        patch("gemma_4_sql.backends.pytorch.etl._load_hf_or_duckdb", return_value="ds"),
        patch("gemma_4_sql.backends.pytorch.etl.SQLTokenizer"),
        patch("gemma_4_sql.backends.pytorch.etl._get_pytorch_classes", return_value=MagicMock()),
        patch("gemma_4_sql.backends.pytorch.etl._get_sampler", return_value=None),
    ):
        res = etl_module.build_dataloader(config)
        assert res["status"] == "loaded"

    monkeypatch.setattr(etl_module, "datasets", None)
    with pytest.raises(DependencyMissingError):
        etl_module.build_dataloader(config)

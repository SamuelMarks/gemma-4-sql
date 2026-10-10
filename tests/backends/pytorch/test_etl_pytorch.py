"""Tests for PyTorch etl."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import ETLConfig


def test_pytorch_etl_imports():
    """Test pytorch etl imports fallback."""
    import importlib

    with patch.dict(sys.modules, {"datasets": None, "torch": None, "torch.utils.data": None}):
        import gemma_4_sql.backends.pytorch.etl as etl_module

        importlib.reload(etl_module)
        assert etl_module.datasets is None
        assert etl_module.torch is None
        assert etl_module.DataLoader is None
        assert etl_module.Dataset is None
    importlib.reload(etl_module)


def test_pytorch_dataset():
    """Test PyTorchDataset."""
    import gemma_4_sql.backends.pytorch.etl as etl_module

    mock_torch = MagicMock()
    mock_torch.long = "long"
    mock_torch.float32 = "float32"
    mock_torch.tensor.side_effect = lambda x, dtype: (x, dtype)
    etl_module.torch = mock_torch

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


def test_collate_fn():
    """Test _collate_fn."""
    import gemma_4_sql.backends.pytorch.etl as etl_module

    mock_torch = MagicMock()
    etl_module.torch = mock_torch
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


def test_get_sampler():
    """Test _get_sampler."""
    import gemma_4_sql.backends.pytorch.etl as etl_module

    assert etl_module._get_sampler("ds", False) is None

    with patch("builtins.__import__") as mock_import:
        mock_dist = MagicMock()
        mock_dist.DistributedSampler.return_value = "sampler"
        mock_import.return_value = mock_dist

        assert etl_module._get_sampler("ds", True) == "sampler"

        mock_dist.DistributedSampler.side_effect = RuntimeError("error")
        assert etl_module._get_sampler("ds", True) is None


def test_load_hf_or_duckdb():
    """Test _load_hf_or_duckdb."""
    import gemma_4_sql.backends.pytorch.etl as etl_module

    with patch("gemma_4_sql.backends.pytorch.etl.load_duckdb_dataset") as mock_load:
        mock_load.return_value = "duck"
        assert etl_module._load_hf_or_duckdb("ds", "split", "path", "table") == "duck"

    mock_datasets = MagicMock()
    etl_module.datasets = mock_datasets
    mock_datasets.load_dataset.return_value = "hf"

    assert etl_module._load_hf_or_duckdb("ds", "split", None, None) == "hf"

    etl_module.datasets = None
    with pytest.raises(DependencyMissingError):
        etl_module._load_hf_or_duckdb("ds", "split", None, None)


def test_build_dataloader():
    """Test build_dataloader."""
    import gemma_4_sql.backends.pytorch.etl as etl_module

    etl_module.datasets = MagicMock()
    etl_module.torch = MagicMock()
    etl_module.Dataset = MagicMock()
    etl_module.DataLoader = MagicMock()

    config = ETLConfig(dataset_name="ds", split="train")

    with patch("gemma_4_sql.backends.pytorch.etl._load_hf_or_duckdb") as mock_load:
        mock_load.return_value = "ds"
        with patch("gemma_4_sql.backends.pytorch.etl.SQLTokenizer"):
            with patch("gemma_4_sql.backends.pytorch.etl._get_pytorch_classes") as mock_cls:
                mock_cls.return_value = MagicMock()
                with patch("gemma_4_sql.backends.pytorch.etl._get_sampler") as mock_samp:
                    mock_samp.return_value = None

                    res = etl_module.build_dataloader(config)
                    assert res["status"] == "loaded"

    etl_module.datasets = None
    with pytest.raises(DependencyMissingError):
        etl_module.build_dataloader(config)

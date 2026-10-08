"""Module docstring."""

from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.backends.pytorch import etl
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import ETLConfig


def test_get_pytorch_classes():
    """Docstring for test_get_pytorch_classes."""
    MagicMock()
    mock_torch = MagicMock()
    mock_torch.tensor.side_effect = lambda x, dtype=None: x
    mock_torch.long = "mock_long"
    mock_torch.float32 = "mock_float32"
    mock_dataset_cls = type("Dataset", (object,), {})

    with patch("gemma_4_sql.backends.pytorch.etl.Dataset", mock_dataset_cls), patch("gemma_4_sql.backends.pytorch.etl.torch", mock_torch):
        PyTorchDataset = etl._get_pytorch_classes()

        mock_hf_ds = [{"sql_prompt": "prompt1", "sql": "query1"}, {"question": "prompt2", "query": "query2", "image_bytes": b"img", "audio_clip": b"aud"}]

        mock_tok = MagicMock()
        mock_tok.encode.side_effect = lambda x: f"encoded_{x}"

        ds = PyTorchDataset(mock_hf_ds, mock_tok)
        assert len(ds) == 2

        # Test item without image/audio
        item1 = ds[0]
        assert "inputs" in item1
        assert "targets" in item1
        assert "pixel_values" not in item1
        assert "audio_values" not in item1

        # Test item with image/audio
        with patch("gemma_4_sql.backends.common_multimodal.process_image") as mock_img, patch("gemma_4_sql.backends.common_multimodal.process_audio") as mock_aud:
            mock_img.return_value = {"pixel_values": "mock_pixel_vals"}
            mock_aud.return_value = {"audio_values": "mock_audio_vals"}

            item2 = ds[1]
            assert item2["pixel_values"] == "mock_pixel_vals"
            assert item2["audio_values"] == "mock_audio_vals"


def test_get_pytorch_classes_no_dataset():
    """Docstring for test_get_pytorch_classes_no_dataset."""
    # If Dataset is None, it should inherit from object
    with patch("gemma_4_sql.backends.pytorch.etl.Dataset", None):
        PyTorchDataset = etl._get_pytorch_classes()
        assert issubclass(PyTorchDataset, object)


def test_collate_fn():
    """Docstring for test_collate_fn."""
    mock_torch = MagicMock()
    mock_torch.nn.utils.rnn.pad_sequence.side_effect = lambda x, batch_first: f"padded_{x}"
    mock_torch.stack.side_effect = lambda x: f"stacked_{x}"

    batch = [
        {"inputs": "i1", "targets": "t1", "pixel_values": "p1", "audio_values": "a1"},
        {"inputs": "i2", "targets": "t2", "pixel_values": "p2", "audio_values": "a2"},
    ]

    with patch("gemma_4_sql.backends.pytorch.etl.torch", mock_torch):
        res = etl._collate_fn(batch)
        assert res["inputs"] == "padded_['i1', 'i2']"
        assert res["targets"] == "padded_['t1', 't2']"
        assert res["pixel_values"] == "stacked_['p1', 'p2']"
        assert res["audio_values"] == "stacked_['a1', 'a2']"

    # Batch without image/audio
    batch2 = [
        {"inputs": "i1", "targets": "t1"},
        {"inputs": "i2", "targets": "t2"},
    ]
    with patch("gemma_4_sql.backends.pytorch.etl.torch", mock_torch):
        res2 = etl._collate_fn(batch2)
        assert "pixel_values" not in res2


def test_get_sampler():
    """Docstring for test_get_sampler."""
    assert etl._get_sampler(None, False) is None

    # Test valid distributed
    class MockDistributedSampler:
        """Docstring for MockDistributedSampler."""

        def __init__(self, ds):
            """Docstring for __init__."""
            self.ds = ds

    with patch("builtins.__import__", return_value=MagicMock(DistributedSampler=MockDistributedSampler)):
        sampler = etl._get_sampler("my_ds", True)
        assert isinstance(sampler, MockDistributedSampler)
        assert sampler.ds == "my_ds"

    # Test import error / value error fallback
    class MockFailingSampler:
        """Docstring for MockFailingSampler."""

        def __init__(self, ds):
            """Docstring for __init__."""
            raise ValueError("mock error")

    with patch("builtins.__import__", return_value=MagicMock(DistributedSampler=MockFailingSampler)):
        assert etl._get_sampler("my_ds", True) is None


def test_load_hf_or_duckdb():
    """Docstring for test_load_hf_or_duckdb."""
    with patch("gemma_4_sql.backends.pytorch.etl.load_duckdb_dataset") as mock_load:
        mock_load.return_value = "duckdb_ds"
        assert etl._load_hf_or_duckdb("ds", "split", "path", "table") == "duckdb_ds"

    # Test missing datasets
    with patch("gemma_4_sql.backends.pytorch.etl.datasets", None), pytest.raises(DependencyMissingError):
        etl._load_hf_or_duckdb("ds", "split", None, None)

    # Test normal hf
    mock_datasets = MagicMock()
    mock_datasets.load_dataset.return_value = "hf_ds"
    with patch("gemma_4_sql.backends.pytorch.etl.datasets", mock_datasets):
        assert etl._load_hf_or_duckdb("ds", "split", None, None) == "hf_ds"


def test_build_dataloader():
    """Docstring for test_build_dataloader."""
    config = ETLConfig(dataset_name="dummy_ds", split="train", batch_size=4, tokenizer_name="dummy_tok", distributed=True, duckdb_path=None, duckdb_table=None)

    # Test missing deps
    with patch("gemma_4_sql.backends.pytorch.etl.datasets", None), pytest.raises(DependencyMissingError):
        etl.build_dataloader(config)

    mock_datasets = MagicMock()
    mock_torch = MagicMock()
    mock_dataset_cls = MagicMock()
    mock_dataloader_cls = MagicMock()

    with (
        patch("gemma_4_sql.backends.pytorch.etl.datasets", mock_datasets),
        patch("gemma_4_sql.backends.pytorch.etl.torch", mock_torch),
        patch("gemma_4_sql.backends.pytorch.etl.Dataset", mock_dataset_cls),
        patch("gemma_4_sql.backends.pytorch.etl.DataLoader", mock_dataloader_cls),
        patch("gemma_4_sql.backends.pytorch.etl._load_hf_or_duckdb", return_value="hf_ds"),
        patch("gemma_4_sql.backends.pytorch.etl.SQLTokenizer"),
        patch("gemma_4_sql.backends.pytorch.etl._get_pytorch_classes") as mock_get_cls,
        patch("gemma_4_sql.backends.pytorch.etl._get_sampler", return_value="sampler"),
    ):
        mock_pt_cls = MagicMock()
        mock_pt_ds = MagicMock()
        mock_pt_cls.return_value = mock_pt_ds
        mock_get_cls.return_value = mock_pt_cls
        mock_dataloader_cls.return_value = "mock_dl"

        res = etl.build_dataloader(config)
        assert res["status"] == "loaded"
        assert res["loader"] == "mock_dl"

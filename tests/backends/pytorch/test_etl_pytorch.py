"""Unit and behavioral tests for PyTorch ETL pipeline with real tensors and SQLTokenizer."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from tests.conftest import requires_torch

pytestmark = requires_torch

import gemma_4_sql.backends.pytorch.etl as etl_module
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.tokenization import SQLTokenizer
from gemma_4_sql.type_hints import ETLConfig


def test_pytorch_dataset_and_collate_real_tensors() -> None:
    """Verify PyTorchDataset and _collate_fn produce real padded torch.Tensor batches."""
    records = [
        {"sql_prompt": "SELECT 1", "sql": "SELECT 1;"},
        {"question": "Find all users", "query": "SELECT * FROM users;", "image_url": "img.png", "audio_clip": "aud.wav"},
    ]
    tok = SQLTokenizer()
    cls = etl_module._get_pytorch_classes()
    dataset = cls(records, tok)
    assert len(dataset) == 2

    with (
        patch("gemma_4_sql.backends.common_multimodal.process_image", return_value={"pixel_values": [[1.0, 2.0]]}),
        patch("gemma_4_sql.backends.common_multimodal.process_audio", return_value={"audio_values": [0.5, -0.5]}),
    ):
        item0 = dataset[0]
        item1 = dataset[1]
        assert isinstance(item0["inputs"], torch.Tensor)
        assert item0["inputs"].dtype == torch.long
        assert isinstance(item1["pixel_values"], torch.Tensor)
        assert item1["pixel_values"].dtype == torch.float32
        assert isinstance(item1["audio_values"], torch.Tensor)

        # Also cover fallback when format_multimodal_prompt returns empty dict
        with patch("gemma_4_sql.backends.common_multimodal.format_multimodal_prompt", return_value={}):
            item1_fallback = dataset[1]
            assert isinstance(item1_fallback["inputs"], torch.Tensor)

        batch_no_mm = etl_module._collate_fn([item0, {"inputs": item1["inputs"], "targets": item1["targets"]}])
        assert batch_no_mm["inputs"].shape[0] == 2
        assert "pixel_values" not in batch_no_mm

        item0_with_mm = dict(item0, pixel_values=item1["pixel_values"], audio_values=item1["audio_values"])
        batch_mm = etl_module._collate_fn([item0_with_mm, item1])
        assert batch_mm["pixel_values"].shape == (2, 1, 2)
        assert batch_mm["audio_values"].shape == (2, 2)


def test_get_sampler_and_load_hf_or_duckdb(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _get_sampler and _load_hf_or_duckdb handle distributed, DuckDB, and HuggingFace sources."""
    assert etl_module._get_sampler("ds", False) is None

    mock_sampler_cls = MagicMock(return_value="sampler")
    monkeypatch.setattr(etl_module, "optional_attr", lambda *_a, **_k: mock_sampler_cls)
    assert etl_module._get_sampler("ds", True) == "sampler"

    mock_sampler_cls.side_effect = RuntimeError("error")
    assert etl_module._get_sampler("ds", True) is None

    monkeypatch.setattr(etl_module, "optional_attr", lambda *_a, **_k: None)
    assert etl_module._get_sampler("ds", True) is None

    with patch("gemma_4_sql.backends.pytorch.etl.load_duckdb_dataset", return_value="duck"):
        assert etl_module._load_hf_or_duckdb("ds", "split", "path", "table") == "duck"

    mock_datasets = MagicMock()
    mock_datasets.load_dataset.return_value = "hf"
    monkeypatch.setattr(etl_module, "datasets", mock_datasets)
    assert etl_module._load_hf_or_duckdb("ds", "split", None, None) == "hf"

    monkeypatch.setattr(etl_module, "datasets", None)
    with pytest.raises(DependencyMissingError):
        etl_module._load_hf_or_duckdb("ds", "split", None, None)


def test_build_dataloader_real_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify build_dataloader constructs a real torch.utils.data.DataLoader yielding padded batches."""
    mock_datasets = MagicMock()
    mock_datasets.load_dataset.return_value = [
        {"sql_prompt": "SELECT 1", "sql": "SELECT 1;"},
        {"sql_prompt": "SELECT 22", "sql": "SELECT 22;"},
    ]
    monkeypatch.setattr(etl_module, "datasets", mock_datasets)

    config = ETLConfig(dataset_name="ds", split="train", batch_size=2)
    res = etl_module.build_dataloader(config)
    assert res["status"] == "loaded"
    loader: Any = res["loader"]
    first_batch = next(iter(loader))
    assert first_batch["inputs"].shape[0] == 2
    assert first_batch["targets"].shape[0] == 2

    monkeypatch.setattr(etl_module, "datasets", None)
    with pytest.raises(DependencyMissingError):
        etl_module.build_dataloader(config)

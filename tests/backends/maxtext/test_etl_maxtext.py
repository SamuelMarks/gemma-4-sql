"""Unit and behavioral tests for MaxText Grain ETL pipeline."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

import gemma_4_sql.backends.maxtext.etl as maxtext_etl
from gemma_4_sql.backends.maxtext.etl import build_dataloader
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import ETLConfig


def test_maxtext_etl_build_dataloader_real_transform(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify maxtext.etl.build_dataloader wraps BaseFormatTransform with MaxText segment_ids and positions."""

    class FakeGrainDataLoader:
        """Lightweight Grain DataLoader substitute executing real MaxTextFormatTransform."""

        def __init__(self, *, data_source: Any, sampler: Any, operations: list[Any], worker_count: int = 0) -> None:
            """Store arguments and transform the first record."""
            self.data_source = data_source
            self.sampler = sampler
            self.first_mapped = operations[0].map(data_source[0])
            self.worker_count = worker_count

    mock_datasets = MagicMock()
    mock_datasets.load_dataset.return_value = [{"sql_prompt": "SELECT 1", "sql": "SELECT 1;"}]
    mock_grain = MagicMock()
    mock_grain.RandomAccessDataSource = object
    mock_grain.MapTransform = object
    mock_grain.DataLoader = FakeGrainDataLoader

    monkeypatch.setattr(maxtext_etl, "datasets", mock_datasets)
    monkeypatch.setattr(maxtext_etl, "grain", mock_grain)

    config = ETLConfig(dataset_name="ds", split="split", batch_size=2, distributed=True, tokenizer_name="tok")
    res = build_dataloader(config)
    assert res["dataset"] == "ds"
    assert res["split"] == "split"
    assert res["status"] == "loaded"
    assert res["batch_size"] == 2
    assert res["backend"] == "maxtext"
    assert res.get("distributed") is True

    loader: Any = res["loader"]
    mapped = loader.first_mapped
    assert "inputs" in mapped
    assert "targets" in mapped
    assert mapped["segment_ids"] == [1]
    assert mapped["positions"] == [0]


def test_build_dataloader_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify build_dataloader raises DependencyMissingError when datasets is None."""
    config = ETLConfig(dataset_name="ds", split="split")
    monkeypatch.setattr(maxtext_etl, "datasets", None)
    with pytest.raises(DependencyMissingError, match=r"Missing grain or datasets\. Cannot load ds\."):
        build_dataloader(config)

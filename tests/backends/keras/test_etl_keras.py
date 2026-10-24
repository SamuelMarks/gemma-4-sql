"""Unit and behavioral tests for Keras Grain ETL pipeline."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from tests.conftest import requires_keras

pytestmark = requires_keras

import pytest

from gemma_4_sql.backends.keras import etl
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import ETLConfig


def test_keras_etl_build_dataloader_real_transform(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify keras.etl.build_dataloader constructs a real HFDataSource and BaseFormatTransform."""

    class FakeGrainDataLoader:
        """Lightweight Grain DataLoader substitute executing real format transformation."""

        def __init__(self, *, data_source: Any, sampler: Any, operations: list[Any], worker_count: int = 0) -> None:
            """Store arguments and transform the first record."""
            self.data_source = data_source
            self.sampler = sampler
            self.first_mapped = operations[0].map(data_source[0])
            self.worker_count = worker_count

    mock_datasets = MagicMock()
    mock_datasets.load_dataset.return_value = [{"sql_prompt": "List tables", "sql": "SELECT name FROM sqlite_master;"}]
    mock_grain = MagicMock()
    mock_grain.RandomAccessDataSource = object
    mock_grain.MapTransform = object
    mock_grain.DataLoader = FakeGrainDataLoader

    monkeypatch.setattr(etl, "datasets", mock_datasets)
    monkeypatch.setattr(etl, "grain", mock_grain)

    cfg = ETLConfig(dataset_name="spider", split="train", batch_size=4, distributed=False)
    res = etl.build_dataloader(cfg)
    assert res["status"] == "loaded"
    assert res["backend"] == "keras"
    assert res["batch_size"] == 4
    loader: Any = res["loader"]
    assert len(loader.data_source) == 1
    assert len(loader.first_mapped["inputs"]) > 0
    assert len(loader.first_mapped["targets"]) > 0


def test_keras_etl_missing_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify keras.etl.build_dataloader raises DependencyMissingError when datasets or grain is None."""
    cfg = ETLConfig(dataset_name="spider", split="train")
    monkeypatch.setattr(etl, "datasets", None)
    with pytest.raises(DependencyMissingError):
        etl.build_dataloader(cfg)

    monkeypatch.setattr(etl, "datasets", MagicMock())
    monkeypatch.setattr(etl, "grain", None)
    with pytest.raises(DependencyMissingError):
        etl.build_dataloader(cfg)

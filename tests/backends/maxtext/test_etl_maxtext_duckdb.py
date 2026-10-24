"""Unit and behavioral tests for etl maxtext duckdb."""

from typing import Any

import pytest

import gemma_4_sql.backends.maxtext.etl as etl_maxtext
from gemma_4_sql.type_hints import ETLConfig


def test_maxtext_etl_duckdb_missing(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Verify build_dataloader raises when duckdb is unavailable."""
    monkeypatch.setattr(etl_maxtext, "duckdb", None, raising=False)
    monkeypatch.setattr(etl_maxtext, "datasets", object())
    monkeypatch.setattr(etl_maxtext, "grain", object())
    with pytest.raises(Exception, match=r".*"):
        etl_maxtext.build_dataloader(ETLConfig(dataset_name="dummy", split="train", batch_size=10, duckdb_path=":memory:", duckdb_table="tbl"))

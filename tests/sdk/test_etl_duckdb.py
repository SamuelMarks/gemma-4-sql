"""Behavioral and unit tests for DuckDB support in ETL pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from gemma_4_sql.backends.common_data import load_duckdb_dataset
from gemma_4_sql.exceptions import ETLError
from gemma_4_sql.sdk.etl import etl_posttrain, etl_pretrain, etl_sft
from gemma_4_sql.type_hints import ETLConfig
from tests.conftest import requires_duckdb

pytestmark = requires_duckdb


@pytest.fixture
def seeded_duckdb_file(tmp_path: Path) -> str:
    """Create a real temporary DuckDB database file populated with SQL training rows."""
    import duckdb

    db_file = str(tmp_path / "etl.duckdb")
    conn = duckdb.connect(db_file)
    conn.execute("CREATE TABLE sql_train (question VARCHAR, query VARCHAR)")
    conn.execute("INSERT INTO sql_train VALUES ('List all users', 'SELECT * FROM users;')")
    conn.execute("INSERT INTO sql_train VALUES ('Count active orders', 'SELECT COUNT(*) FROM orders WHERE active = 1;')")
    conn.close()
    return db_file


def test_real_duckdb_etl_stages(seeded_duckdb_file: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify etl_pretrain, etl_sft, and etl_posttrain load real rows from a DuckDB file across backends."""
    rows = load_duckdb_dataset(seeded_duckdb_file, "sql_train")
    assert len(rows) == 2
    assert rows[0] == {"question": "List all users", "query": "SELECT * FROM users;"}
    assert rows[1] == {"question": "Count active orders", "query": "SELECT COUNT(*) FROM orders WHERE active = 1;"}

    # MLX backend uses pure Python MLXDataLoader over the real DuckDB rows
    monkeypatch.setattr("gemma_4_sql.backends.mlx.etl.datasets", object(), raising=False)
    config = ETLConfig(
        dataset_name="local_duckdb",
        split="train",
        batch_size=2,
        duckdb_path=seeded_duckdb_file,
        duckdb_table="sql_train",
    )

    res_pre = etl_pretrain(config, backend="mlx")
    assert res_pre["status"] == "loaded"
    assert res_pre["backend"] == "mlx"
    loader_pre: Any = res_pre.get("loader")
    batches_pre = list(loader_pre)
    assert len(batches_pre) == 1
    assert len(batches_pre[0]["inputs"]) == 2
    assert len(batches_pre[0]["targets"]) == 2

    res_sft = etl_sft(config, backend="mlx")
    assert res_sft["status"] == "loaded"
    batches_sft = list(res_sft.get("loader") or ())
    assert len(batches_sft) == 1

    res_post = etl_posttrain(config, backend="mlx")
    assert res_post["status"] == "loaded"
    batches_post = list(res_post.get("loader") or ())
    assert len(batches_post) == 1


@pytest.mark.parametrize("backend", ["jax", "keras", "maxtext"])
def test_duckdb_etl_grain_backends(
    seeded_duckdb_file: str,
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Grain-based backends (jax, keras, maxtext) load real rows from DuckDB."""
    mock_grain = MagicMock()
    mock_grain.RandomAccessDataSource = object
    mock_grain.MapTransform = object
    monkeypatch.setattr(f"gemma_4_sql.backends.{backend}.etl.grain", mock_grain, raising=False)
    monkeypatch.setattr(f"gemma_4_sql.backends.{backend}.etl.datasets", object(), raising=False)

    config = ETLConfig(
        dataset_name="local_duckdb",
        split="train",
        duckdb_path=seeded_duckdb_file,
        duckdb_table="sql_train",
    )
    res = etl_pretrain(config, backend=backend)
    assert res["status"] == "loaded"
    assert res["backend"] == backend
    call_kwargs = mock_grain.DataLoader.call_args.kwargs
    source = call_kwargs["data_source"]
    assert len(source) == 2
    assert source[0] == {"question": "List all users", "query": "SELECT * FROM users;"}


def test_duckdb_etl_invalid_table_raises(seeded_duckdb_file: str) -> None:
    """Verify load_duckdb_dataset rejects unsafe SQL injection table names and non-existent tables."""
    with pytest.raises(ETLError, match="Invalid or unsafe table name"):
        load_duckdb_dataset(seeded_duckdb_file, "sql_train; DROP TABLE sql_train")

    with pytest.raises(ETLError, match="DuckDB error"):
        load_duckdb_dataset(seeded_duckdb_file, "nonexistent_table")

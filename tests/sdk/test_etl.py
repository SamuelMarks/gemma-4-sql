"""Behavioral tests for the SDK ETL module across backends and DuckDB."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from gemma_4_sql.constants import DEFAULT_POSTTRAIN_DATASET, DEFAULT_SFT_DATASET
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.sdk import etl
from gemma_4_sql.sdk.etl import _route_backend, etl_posttrain, etl_pretrain, etl_sft
from gemma_4_sql.sdk.registry import get_backend
from gemma_4_sql.type_hints import ETLConfig
from tests.conftest import requires_duckdb


def test_route_backend_invalid() -> None:
    """Verify _route_backend raises ValueError on an unknown backend."""
    with pytest.raises(ValueError, match="Unknown backend: invalid"):
        _route_backend(ETLConfig(dataset_name="dummy/data", split="train", batch_size=16, distributed=False), "invalid")


@requires_duckdb
def test_etl_real_duckdb_execution_mlx(tmp_path: Path) -> None:
    """Verify etl_sft builds a real tokenized dataloader from a DuckDB table in tmp_path."""
    import duckdb

    db_file = tmp_path / "etl_sdk_test.duckdb"
    conn = duckdb.connect(str(db_file))
    conn.execute("CREATE TABLE sql_pairs (question VARCHAR, sql VARCHAR);")
    conn.execute("INSERT INTO sql_pairs VALUES ('Count users', 'SELECT count(*) FROM users;');")
    conn.execute("INSERT INTO sql_pairs VALUES ('List names', 'SELECT name FROM users;');")
    conn.close()

    cfg = ETLConfig(
        dataset_name="local_duckdb",
        split="train",
        batch_size=2,
        distributed=False,
        duckdb_path=str(db_file),
        duckdb_table="sql_pairs",
    )
    res = etl_sft(cfg, backend="mlx")
    assert res["backend"] == "mlx"
    assert res["dataset"] == "local_duckdb"
    assert res["batch_size"] == 2
    assert res["status"] == "loaded"
    loader = res["loader"]
    batches = list(loader)  # type: ignore[arg-type]
    assert len(batches) == 1
    assert "inputs" in batches[0]
    assert "targets" in batches[0]


def test_route_backend_pytorch_forwarding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _route_backend forwards exact ETLConfig fields to the PyTorch backend."""
    pytorch_agent = get_backend("pytorch")
    captured: dict[str, Any] = {}

    def spy_build(config: ETLConfig, **kwargs: Any) -> dict[str, Any]:
        """Record forwarded ETLConfig."""
        captured["config"] = config
        captured["kwargs"] = kwargs
        return {"backend": "pytorch", "dataset": config.dataset_name, "batch_size": config.batch_size}

    monkeypatch.setattr(pytorch_agent, "build_dataloader", spy_build)
    cfg = ETLConfig(dataset_name="dummy/data", split="train", batch_size=16, distributed=False)
    res = _route_backend(cfg, "pytorch")
    assert res["backend"] == "pytorch"
    assert captured["config"] == cfg


def test_etl_pretrain_propagates_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify etl_pretrain propagates DependencyMissingError when backend dependencies are missing."""
    config = ETLConfig(dataset_name="dummy/data", split="train", batch_size=16, distributed=False)
    jax_agent = get_backend("jax")

    def raise_err(*_a: Any, **_k: Any) -> None:
        """Raise DependencyMissingError."""
        raise DependencyMissingError("Mocked missing JAX")

    monkeypatch.setattr(jax_agent, "build_dataloader", raise_err)
    with pytest.raises(DependencyMissingError, match="Mocked missing JAX"):
        etl_pretrain(config, backend="jax")


def test_etl_sft_and_posttrain_forwarding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify etl_sft and etl_posttrain forward ETLConfig to the requested backend."""
    config = ETLConfig(dataset_name="dummy/data", split="train", batch_size=16, distributed=False)
    keras_agent = get_backend("keras")
    monkeypatch.setattr(
        keras_agent,
        "build_dataloader",
        lambda cfg, **_k: {"backend": "keras", "dataset": cfg.dataset_name, "batch_size": cfg.batch_size},
    )
    res_sft = etl_sft(config, backend="keras")
    assert res_sft["dataset"] == "dummy/data"
    assert res_sft["batch_size"] == 16

    maxtext_agent = get_backend("maxtext")
    monkeypatch.setattr(
        maxtext_agent,
        "build_dataloader",
        lambda cfg, **_k: {"backend": "maxtext", "dataset": cfg.dataset_name, "batch_size": cfg.batch_size},
    )
    res_post = etl_posttrain(config, backend="maxtext")
    assert res_post["dataset"] == "dummy/data"
    assert res_post["batch_size"] == 16


def test_etl_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify etl_pretrain, etl_sft, and etl_posttrain populate default ETLConfig when None is passed."""
    captured: list[tuple[ETLConfig, str]] = []

    def spy_route(c: ETLConfig, b: str) -> dict[str, Any]:
        """Record routed config and backend."""
        captured.append((c, b))
        return {"dataset": c.dataset_name, "backend": b}

    monkeypatch.setattr(etl, "_route_backend", spy_route)

    assert etl.etl_pretrain(ETLConfig(dataset_name="d", split="train"))["dataset"] == "d"
    assert etl.etl_pretrain(None)["dataset"] == "my-custom-dataset"

    assert etl.etl_sft(ETLConfig(dataset_name="d2", split="train"))["dataset"] == "d2"
    assert etl.etl_sft(None)["dataset"] == DEFAULT_SFT_DATASET

    assert etl.etl_posttrain(ETLConfig(dataset_name="d3", split="train"))["dataset"] == "d3"
    assert etl.etl_posttrain(None)["dataset"] == DEFAULT_POSTTRAIN_DATASET
    assert len(captured) == 6
    assert [c.stage for c, _ in captured] == ["pretrain", "pretrain", "sft", "sft", "posttrain", "posttrain"]

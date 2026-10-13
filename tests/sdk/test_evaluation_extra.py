"""Module docstring."""

import pytest

from gemma_4_sql.sdk.db_engine import LiveDatabaseEngine
from gemma_4_sql.sdk.evaluation import _process_batch_inputs, compute_metrics_async


def test_extract_batch_ids_dict_missing_keys():
    """Docstring for test_extract_batch_ids_dict_missing_keys."""
    inputs, targets = _process_batch_inputs({"foo": "bar"})
    assert inputs == []
    assert targets == []


def test_extract_batch_ids_dict_no_getitem():
    """Docstring for test_extract_batch_ids_dict_no_getitem."""
    inputs, targets = _process_batch_inputs({"inputs": 123, "targets": 456})
    assert inputs == []
    assert targets == []


@pytest.mark.asyncio
async def test_compute_metrics_async_mismatch():
    """Docstring for test_compute_metrics_async_mismatch."""
    engine = LiveDatabaseEngine(":memory:")
    with pytest.raises(ValueError, match="Length mismatch"):
        await compute_metrics_async(engine, ["a", "b"], ["a"])


@pytest.mark.asyncio
async def test_compute_metrics_async_exception(monkeypatch):
    """Docstring for test_compute_metrics_async_exception."""
    engine = LiveDatabaseEngine(":memory:")

    async def mock_exec(query):
        raise RuntimeError("Mock error")

    monkeypatch.setattr(engine, "execute_with_feedback_async", mock_exec)
    metrics = await compute_metrics_async(engine, ["a"], ["a"])
    assert metrics["valid_sql"] == 0.0
    assert metrics["execution_accuracy"] == 0.0

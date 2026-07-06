"""Extra tests for SDK coverage."""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from gemma_4_sql.sdk.adapters.base import DatabaseAdapter
from gemma_4_sql.sdk.agent import AgentContext, _process_single_prompt
from gemma_4_sql.sdk.db_engine import LiveDatabaseEngine
from gemma_4_sql.sdk.evaluation import _run_evaluation_inference, compute_metrics, evaluate
from gemma_4_sql.sdk.rag import retrieve_relevant_schema


def test_db_engine_connect_close() -> None:
    """Test db engine connect and close."""
    with patch("gemma_4_sql.sdk.db_engine._ADAPTERS") as mock_adapters:
        mock_adapter_cls = MagicMock()
        mock_adapter = MagicMock()
        mock_adapter.connect.return_value = "conn"
        mock_adapter_cls.return_value = mock_adapter
        mock_adapters.get.return_value = mock_adapter_cls
        engine = LiveDatabaseEngine(db_type="sqlite", db_path=":memory:")
        assert engine.connect() == "conn"
        engine.close()
        mock_adapter.close.assert_called_once()


def test_db_engine_unsupported_type() -> None:
    """Test unsupported db type."""
    with pytest.raises(ValueError, match="Unsupported db_type"):
        LiveDatabaseEngine(db_type="invalid_type")


def test_db_engine_ddl() -> None:
    """Test db engine ddl."""
    with patch("gemma_4_sql.sdk.db_engine._ADAPTERS") as mock_adapters:
        mock_adapter_cls = MagicMock()
        mock_adapter = MagicMock()
        mock_adapter_cls.return_value = mock_adapter
        mock_adapters.get.return_value = mock_adapter_cls
        LiveDatabaseEngine(ddl="CREATE TABLE t (a INT);")
        mock_adapter.setup_schema.side_effect = RuntimeError("error")
        with pytest.raises(RuntimeError):
            LiveDatabaseEngine(ddl="CREATE TABLE t (a INT);")


def test_db_engine_compare_queries() -> None:
    """Test compare queries."""
    with patch("gemma_4_sql.sdk.db_engine._ADAPTERS") as mock_adapters:
        mock_adapter_cls = MagicMock()
        mock_adapter = MagicMock()
        mock_adapter.execute_query.side_effect = [[(1,)], [(1,)], [(1,)], [(2,)]]
        mock_adapter_cls.return_value = mock_adapter
        mock_adapters.get.return_value = mock_adapter_cls
        engine = LiveDatabaseEngine()
        assert engine.compare_queries("q1", "q2") is True
        assert engine.compare_queries("q1", "q2") is False


@pytest.mark.asyncio
async def test_db_engine_execute_async() -> None:
    """Test db engine execute async."""
    with patch("gemma_4_sql.sdk.db_engine._ADAPTERS") as mock_adapters:
        mock_adapter_cls = MagicMock()
        mock_adapter = MagicMock()
        mock_adapter.execute_query_async = AsyncMock(return_value=[("row",)])
        mock_adapter.execute_with_feedback_async = AsyncMock(return_value=(True, [("row",)], None))
        mock_adapter_cls.return_value = mock_adapter
        mock_adapters.get.return_value = mock_adapter_cls
        engine = LiveDatabaseEngine()
        assert await engine.execute_query_async("q") == [("row",)]
        assert await engine.execute_with_feedback_async("q") == (True, [("row",)], None)


@pytest.mark.asyncio
async def test_process_single_prompt_coverage() -> None:
    """Test process single prompt coverage."""
    backend_impl = MagicMock()
    backend_impl.generate_sql.side_effect = [{"sql": "SELECT 1", "confidence_score": 0.4}, {"sql": "INVALID", "confidence_score": 0.9}, {"sql": "SELECT 1", "confidence_score": 0.9}]
    engine = MagicMock()
    engine.execute_with_feedback_async = AsyncMock(side_effect=[(False, [], "syntax error"), (True, [(1,)], None)])
    ctx = AgentContext(min_confidence=0.5)
    res = await _process_single_prompt("jax", backend_impl, "model", "prompt", engine, ctx)
    assert res["success"] is True


def test_compute_metrics() -> None:
    """Test compute metrics."""
    engine = MagicMock()

    async def mock_execute(*args: object, **kwargs: object) -> tuple[bool, list, None]:
        """Docstring."""
        return True, [(1,)], None

    engine.execute_with_feedback_async.side_effect = mock_execute

    async def mock_compare(*args: object, **kwargs: object) -> bool:
        """Docstring."""
        return True

    engine.compare_queries_async.side_effect = mock_compare
    res = compute_metrics(engine, ["SELECT 1"], ["SELECT 1"])
    assert res["exact_match"] == 1.0


def test_evaluate_mock_predictions() -> None:
    """Test evaluate with mock predictions."""
    with patch("gemma_4_sql.sdk.evaluation.compute_metrics_async") as mock_compute:

        async def mock_compute_async(*args: object, **kwargs: object) -> dict:
            """Docstring."""
            return {"exact_match": 1.0}

        mock_compute.side_effect = mock_compute_async
        with patch("gemma_4_sql.sdk.registry.get_backend"):
            res = evaluate("model", "dataset", mock_predictions=["s1"], mock_truths=["s1"])
            assert res["metrics"]["exact_match"] == 1.0


def test_run_evaluation_inference_no_dataloader() -> None:
    """Test run evaluation inference no dataloader."""
    backend_impl = MagicMock()
    backend_impl.build_dataloader.return_value = {"loader": None}
    backend_impl.generate_sql.return_value = {"sql": "SELECT 2"}
    preds, _truths, _scores = _run_evaluation_inference("model", "dataset", backend_impl)
    assert preds[0] == "SELECT 2"


def test_rag_semantic_no_relevant() -> None:
    """Test rag semantic no relevant."""
    mock_st = MagicMock()
    mock_st.return_value.encode.return_value = np.array([[1.0]])

    with patch("gemma_4_sql.sdk.rag.SentenceTransformer", mock_st), patch("gemma_4_sql.sdk.rag.cosine_similarity", return_value=np.array([[0.0, 0.0]])):
        schema = {"t1": ["c1"], "t2": ["c2"]}
        res = retrieve_relevant_schema("prompt", schema)
        assert "Table: t1" in res
        assert "Table: t2" in res


def test_base_methods() -> None:
    """Test base methods."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class DatabaseAdapter"):
        DatabaseAdapter()

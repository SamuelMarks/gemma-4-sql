"""Tests for SDK Evaluation module."""

import pytest

from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.sdk.evaluation import evaluate


def test_evaluate_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test evaluate with jax."""
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    jax_agent = get_backend("jax")
    monkeypatch.setattr(jax_agent, "generate_sql", lambda *_args, **_kwargs: {"sql": "SELECT 1"})

    def raise_err(*a: object, **k: object) -> None:
        """Mock error when building dataloader."""
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("Mocked missing JAX")

    monkeypatch.setattr(jax_agent, "build_dataloader", raise_err)
    with pytest.raises(DependencyMissingError):
        evaluate("model1", "data1", "jax")


def test_evaluate_keras(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test evaluate with keras.

    Raises:
        AssertionError: Description.

    """
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    keras_agent = get_backend("keras")
    monkeypatch.setattr(keras_agent, "build_dataloader", lambda *args, **kwargs: {"loader": [{"inputs": [[1]], "targets": [[1]]}]})
    monkeypatch.setattr(keras_agent, "generate_sql", lambda *_args, **_kwargs: {"sql": "SELECT 1"})
    res = evaluate("model1", "data1", "keras")
    if res["backend"] != "keras":
        raise AssertionError
    if res["model"] != "model1":
        raise AssertionError
    if res["dataset"] != "data1":
        raise AssertionError
    if "metrics" not in res:
        raise AssertionError


def test_evaluate_maxtext(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test evaluate with maxtext.

    Raises:
        AssertionError: Description.

    """
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    maxtext_agent = get_backend("maxtext")
    monkeypatch.setattr(maxtext_agent, "generate_sql", lambda *_args, **_kwargs: {"sql": "SELECT 1"})
    monkeypatch.setattr(maxtext_agent, "build_dataloader", lambda *_args, **_kwargs: {"loader": [{"inputs": [[1]], "targets": [[2]]}]})
    res = evaluate("model1", "data1", "maxtext")
    if res["backend"] != "maxtext":
        raise AssertionError
    if res["model"] != "model1":
        raise AssertionError
    if res["dataset"] != "data1":
        raise AssertionError
    if "metrics" not in res:
        raise AssertionError


def test_evaluate_pytorch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test evaluate with pytorch.

    Raises:
        AssertionError: Description.

    """
    get_backend = __import__("gemma_4_sql.sdk.registry", fromlist=["get_backend"]).get_backend
    pytorch_agent = get_backend("pytorch")
    monkeypatch.setattr(pytorch_agent, "generate_sql", lambda *_args, **_kwargs: {"sql": "SELECT 1"})
    monkeypatch.setattr(pytorch_agent, "build_dataloader", lambda *_args, **_kwargs: {"loader": [([[1]], [[2]])]})
    res = evaluate("model1", "data1", "pytorch")
    if res["backend"] != "pytorch":
        raise AssertionError
    if res["model"] != "model1":
        raise AssertionError
    if res["dataset"] != "data1":
        raise AssertionError
    if "metrics" not in res:
        raise AssertionError


@pytest.mark.usefixtures("monkeypatch")
def test_evaluate_invalid() -> None:
    """Test evaluate with invalid backend."""
    with pytest.raises(ValueError, match="Unknown backend: invalid"):
        evaluate("model1", "data1", "invalid")


from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.sdk.evaluation import _run_evaluation_inference, compute_metrics


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
    assert res["exact_match"] == pytest.approx(1.0)


def test_evaluate_mock_predictions() -> None:
    """Test evaluate with mock predictions."""
    with patch("gemma_4_sql.sdk.evaluation.compute_metrics_async") as mock_compute:

        async def mock_compute_async(*args: object, **kwargs: object) -> dict:
            """Docstring."""
            return {"exact_match": 1.0}

        mock_compute.side_effect = mock_compute_async
        with patch("gemma_4_sql.sdk.registry.get_backend"):
            res = evaluate("model", "dataset", mock_predictions=["s1"], mock_truths=["s1"])
            assert res["metrics"]["exact_match"] == pytest.approx(1.0)


def test_run_evaluation_inference_no_dataloader() -> None:
    """Test run evaluation inference no dataloader."""
    backend_impl = MagicMock()
    backend_impl.build_dataloader.return_value = {"loader": None}
    backend_impl.generate_sql.return_value = {"sql": "SELECT 2"}
    preds, _truths, _scores = _run_evaluation_inference("model", "dataset", backend_impl)
    assert preds[0] == "SELECT 2"


@pytest.mark.asyncio
async def test_negative_execution_accuracy_syntax_error() -> None:
    """Test compute_metrics_async when predicted SQL has a syntax error.

    Returns:
        None.
    """
    from gemma_4_sql.sdk.db_engine import LiveDatabaseEngine
    from gemma_4_sql.sdk.evaluation import compute_metrics_async

    engine = LiveDatabaseEngine(":memory:", db_type="sqlite")
    preds = ["SELEKT INVALID SQL FROM tbl"]
    truths = ["SELECT 1"]

    metrics = await compute_metrics_async(engine, preds, truths)
    assert metrics["valid_sql"] == pytest.approx(0.0)
    assert metrics["execution_accuracy"] == pytest.approx(0.0)
    assert metrics["exact_match"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_negative_execution_accuracy_both_invalid() -> None:
    """Test compute_metrics_async when both predicted and truth SQL queries are invalid.

    Returns:
        None.
    """
    from gemma_4_sql.sdk.db_engine import LiveDatabaseEngine
    from gemma_4_sql.sdk.evaluation import compute_metrics_async

    engine = LiveDatabaseEngine(":memory:", db_type="sqlite")
    preds = ["BROKEN QUERY ONE"]
    truths = ["BROKEN QUERY TWO"]

    metrics = await compute_metrics_async(engine, preds, truths)
    assert metrics["valid_sql"] == pytest.approx(0.0)
    assert metrics["execution_accuracy"] == pytest.approx(0.0)
    assert metrics["exact_match"] == pytest.approx(0.0)


def test_extract_batch_ids_fallback() -> None:
    """Test _process_batch_inputs with unexpected batch object.

    Returns:
        None.
    """
    from gemma_4_sql.sdk.evaluation import _process_batch_inputs

    inputs, targets = _process_batch_inputs(None)
    assert inputs == []
    assert targets == []


def test_evaluate_empty_confidence_scores() -> None:
    """Test evaluate with empty mock predictions to cover confidence_scores False branch.

    Returns:
        None.
    """
    from gemma_4_sql.sdk.evaluation import evaluate

    res = evaluate("model", "dataset", "jax", mock_predictions=[], mock_truths=[])
    assert res["status"] == "completed"
    assert "mean_confidence" not in res["metrics"]

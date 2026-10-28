"""Behavioral unit and integration tests for the SDK Evaluation module."""

from __future__ import annotations

from collections.abc import Callable
from typing import NoReturn
from unittest.mock import MagicMock, patch

import pytest

from gemma_4_sql.exceptions import DependencyMissingError, EvaluationError
from gemma_4_sql.sdk.db_engine import LiveDatabaseEngine
from gemma_4_sql.sdk.evaluation import (
    _process_batch_inputs,
    _run_evaluation_inference,
    compute_metrics,
    compute_metrics_async,
    evaluate,
    normalize_sql,
)
from gemma_4_sql.sdk.registry import get_backend
from gemma_4_sql.tokenization import SQLTokenizer
from gemma_4_sql.type_hints import ETLConfig, InferenceResultDict, JSONPrimitive, JSONValue


def test_evaluate_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify evaluate propagates DependencyMissingError when JAX dataloader dependencies are missing."""
    jax_agent = get_backend("jax")
    monkeypatch.setattr(jax_agent, "generate_sql", lambda *_args, **_kwargs: {"sql": "SELECT 1"})  # pyright: ignore[reportUnknownLambdaType]

    def raise_err(*_a: ETLConfig, **_k: JSONValue) -> NoReturn:
        """Raise DependencyMissingError when building dataloader."""
        raise DependencyMissingError("Mocked missing JAX", backend="jax")

    monkeypatch.setattr(jax_agent, "build_dataloader", raise_err)
    with pytest.raises(DependencyMissingError, match="Mocked missing JAX"):
        evaluate("model1", "data1", "jax")


def test_evaluate_keras_with_live_sqlite(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify end-to-end evaluation with Keras backend against a seeded in-memory SQLite database."""
    tokenizer = SQLTokenizer(model_name=None)
    prompt_str = "Get user 1 id and name"
    target_sql = "SELECT id, name FROM users WHERE id = 1"
    encoded_target = tokenizer.encode(f"{prompt_str} {target_sql}")
    encoded_prompt = tokenizer.encode(prompt_str)

    keras_agent = get_backend("keras")
    monkeypatch.setattr(
        keras_agent,
        "build_dataloader",
        lambda *_args, **_kwargs: {"loader": [{"inputs": [encoded_prompt], "targets": [encoded_target]}]},  # pyright: ignore[reportUnknownLambdaType]
    )
    # Semantically and structurally equivalent query with extra whitespace, comma spacing, and trailing semicolon
    monkeypatch.setattr(
        keras_agent,
        "generate_sql",
        lambda *_args, **_kwargs: {"sql": "  select  id , name  FROM users  where  id = 1 ; ", "confidence_score": 0.92},  # pyright: ignore[reportUnknownLambdaType]
    )
    ddl = "CREATE TABLE users (id INT, name TEXT); INSERT INTO users VALUES (1, 'Alice'); INSERT INTO users VALUES (2, 'Bob');"
    res = evaluate("model1", "data1", "keras", ddl=ddl)
    assert res["backend"] == "keras"
    assert res["model"] == "model1"
    assert res["dataset"] == "data1"
    assert res["metrics"]["exact_match"] == pytest.approx(1.0)
    assert res["metrics"]["valid_sql"] == pytest.approx(1.0)
    assert res["metrics"]["execution_accuracy"] == pytest.approx(1.0)
    assert res["metrics"].get("mean_confidence") == pytest.approx(0.92)


def test_evaluate_maxtext(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify evaluate with MaxText backend computes exact match and execution accuracy."""
    maxtext_agent = get_backend("maxtext")
    monkeypatch.setattr(maxtext_agent, "generate_sql", lambda *_args, **_kwargs: {"sql": "SELECT 1"})  # pyright: ignore[reportUnknownLambdaType]
    monkeypatch.setattr(
        maxtext_agent,
        "build_dataloader",
        lambda *_args, **_kwargs: {"loader": [{"inputs": [[1]], "targets": [[2]]}]},  # pyright: ignore[reportUnknownLambdaType]
    )
    res = evaluate("model1", "data1", "maxtext")
    assert res["backend"] == "maxtext"
    assert res["model"] == "model1"
    assert res["dataset"] == "data1"
    assert res["metrics"]["valid_sql"] == pytest.approx(1.0)


def test_evaluate_pytorch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify evaluate with PyTorch backend handles tuple batches from DataLoader."""
    pytorch_agent = get_backend("pytorch")
    monkeypatch.setattr(pytorch_agent, "generate_sql", lambda *_args, **_kwargs: {"sql": "SELECT 1"})  # pyright: ignore[reportUnknownLambdaType]
    monkeypatch.setattr(
        pytorch_agent,
        "build_dataloader",
        lambda *_args, **_kwargs: {"loader": [([[1]], [[2]])]},  # pyright: ignore[reportUnknownLambdaType]
    )
    res = evaluate("model1", "data1", "pytorch")
    assert res["backend"] == "pytorch"
    assert res["model"] == "model1"
    assert res["dataset"] == "data1"
    assert res["metrics"]["valid_sql"] == pytest.approx(1.0)


def test_evaluate_invalid() -> None:
    """Verify evaluate raises ValueError on an unknown backend identifier."""
    with pytest.raises(ValueError, match="Unknown backend: invalid"):
        evaluate("model1", "data1", "invalid")


def test_compute_metrics_with_live_sqlite() -> None:
    """Verify compute_metrics distinguishes exact match, execution equivalence, and wrong results."""
    ddl = "CREATE TABLE users (id INT, name TEXT); INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob');"
    engine = LiveDatabaseEngine(":memory:", ddl=ddl, db_type="sqlite")
    try:
        preds = [
            "SELECT id, name FROM users",  # Exact match & execution match
            "SELECT id, name FROM users WHERE 1=1",  # Not exact match, but execution match!
            "SELECT name, id FROM users",  # Valid SQL, but different column order -> execution mismatch
        ]
        truths = [
            "SELECT id, name FROM users;",
            "SELECT id, name FROM users",
            "SELECT id, name FROM users",
        ]
        res = compute_metrics(engine, preds, truths)
        assert res["exact_match"] == pytest.approx(1.0 / 3.0)
        assert res["valid_sql"] == pytest.approx(1.0)
        assert res["execution_accuracy"] == pytest.approx(2.0 / 3.0)
    finally:
        engine.close()


def test_evaluate_end_to_end_and_engine_close_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify evaluate runs end-to-end with real compute_metrics and closes LiveDatabaseEngine if inference raises."""
    tokenizer = SQLTokenizer(model_name=None)
    encoded_prompt = tokenizer.encode("Count users")
    encoded_truth = tokenizer.encode("SELECT COUNT(*) FROM users")

    jax_agent = get_backend("jax")
    monkeypatch.setattr(
        jax_agent,
        "build_dataloader",
        lambda *_a, **_k: {"loader": [{"inputs": [encoded_prompt], "targets": [encoded_truth]}]},  # pyright: ignore[reportUnknownLambdaType]
    )
    monkeypatch.setattr(
        jax_agent,
        "generate_sql",
        lambda *_a, **_k: {"sql": "SELECT COUNT(*) FROM users", "confidence_score": 0.99, "status": "success"},  # pyright: ignore[reportUnknownLambdaType]
    )
    res = evaluate(
        "model",
        "dataset",
        backend="jax",
        ddl="CREATE TABLE users (id INT); INSERT INTO users VALUES (1), (2);",
    )
    assert res["metrics"]["exact_match"] == pytest.approx(1.0)
    assert res["metrics"]["execution_accuracy"] == pytest.approx(1.0)
    assert res["metrics"].get("mean_confidence") == pytest.approx(0.99)

    closed = False
    orig_close = LiveDatabaseEngine.close

    def tracking_close(self: LiveDatabaseEngine) -> None:
        """Track engine close on exception."""
        nonlocal closed
        closed = True
        orig_close(self)

    monkeypatch.setattr(LiveDatabaseEngine, "close", tracking_close)
    monkeypatch.setattr(jax_agent, "build_dataloader", lambda *_a, **_k: {"loader": None})  # pyright: ignore[reportUnknownLambdaType]
    with pytest.raises(EvaluationError, match="Valid iterable dataloader is required"):
        evaluate("model", "dataset", backend="jax")
    assert closed is True


@pytest.mark.asyncio
async def test_compute_metrics_inside_running_event_loop() -> None:
    """Verify synchronous compute_metrics succeeds when invoked inside an active asyncio event loop."""
    with LiveDatabaseEngine(":memory:", ddl="CREATE TABLE t (x INT); INSERT INTO t VALUES (7);") as engine:
        metrics = compute_metrics(engine, ["SELECT x FROM t"], ["SELECT x FROM t"])
        assert metrics["exact_match"] == pytest.approx(1.0)
        assert metrics["execution_accuracy"] == pytest.approx(1.0)


def test_run_evaluation_inference_no_dataloader() -> None:
    """Verify _run_evaluation_inference raises EvaluationError when dataloader is None or backend fails."""
    backend_impl = MagicMock()
    backend_impl.build_dataloader.return_value = {"loader": None}
    with pytest.raises(EvaluationError, match="Valid iterable dataloader is required for evaluation"):
        _run_evaluation_inference("model", "dataset", backend_impl)

    backend_impl.build_dataloader.return_value = {"loader": [{"inputs": [[65]], "targets": [[65]]}]}
    backend_impl.generate_sql.return_value = {"sql": "SELECT 1", "confidence_score": 0.9, "status": "failed: boom"}
    preds, _truths, scores = _run_evaluation_inference("model", "dataset", backend_impl)
    assert preds == [""]
    assert scores == [0.0]


@pytest.mark.asyncio
async def test_negative_execution_accuracy_syntax_error() -> None:
    """Verify compute_metrics_async records 0.0 valid_sql and execution_accuracy on syntax errors."""
    engine = LiveDatabaseEngine(":memory:", db_type="sqlite")
    try:
        preds = ["SELEKT INVALID SQL FROM tbl"]
        truths = ["SELECT 1"]

        metrics = await compute_metrics_async(engine, preds, truths)
        assert metrics["valid_sql"] == pytest.approx(0.0)
        assert metrics["execution_accuracy"] == pytest.approx(0.0)
        assert metrics["exact_match"] == pytest.approx(0.0)
    finally:
        engine.close()


@pytest.mark.asyncio
async def test_negative_execution_accuracy_both_invalid() -> None:
    """Verify compute_metrics_async records 0.0 execution_accuracy when both queries are invalid."""
    engine = LiveDatabaseEngine(":memory:", db_type="sqlite")
    try:
        preds = ["BROKEN QUERY ONE"]
        truths = ["BROKEN QUERY TWO"]

        metrics = await compute_metrics_async(engine, preds, truths)
        assert metrics["valid_sql"] == pytest.approx(0.0)
        assert metrics["execution_accuracy"] == pytest.approx(0.0)
        assert metrics["exact_match"] == pytest.approx(0.0)
    finally:
        engine.close()


def test_extract_batch_ids_fallback() -> None:
    """Verify _process_batch_inputs returns empty lists for unsupported batch types."""
    inputs, targets = _process_batch_inputs(None)
    assert inputs == []
    assert targets == []


def test_evaluate_empty_confidence_scores() -> None:
    """Verify evaluate omits mean_confidence when no predictions are generated."""
    with (
        patch("gemma_4_sql.sdk.registry.get_backend"),
        patch("gemma_4_sql.sdk.evaluation._run_evaluation_inference", return_value=([], [], [])),
    ):
        res = evaluate("model", "dataset", "jax")
        assert res["status"] == "completed"
        assert "mean_confidence" not in res["metrics"]


def test_run_evaluation_inference_max_batches() -> None:
    """Verify _run_evaluation_inference respects MAX_BATCHES and custom max_batches."""
    backend_impl = MagicMock()
    mock_loader = [({"inputs": [[1]], "targets": [[1]]}) for _ in range(15)]
    backend_impl.build_dataloader.return_value = {"loader": mock_loader}
    backend_impl.generate_sql.return_value = {"sql": "SELECT 1", "confidence_score": 0.9}

    preds, truths, scores = _run_evaluation_inference("model", "dataset", backend_impl)
    assert len(preds) == 10
    assert len(truths) == 10
    assert len(scores) == 10

    preds_custom, _, _ = _run_evaluation_inference("model", "dataset", backend_impl, max_batches=4)
    assert len(preds_custom) == 4

    preds_all, _, _ = _run_evaluation_inference("model", "dataset", backend_impl, max_batches=0)
    assert len(preds_all) == 15


def test_evaluate_with_db_kwargs_and_ddl() -> None:
    """Verify evaluate initializes schema DDL and executes queries against the live engine."""
    with (
        patch("gemma_4_sql.sdk.registry.get_backend"),
        patch(
            "gemma_4_sql.sdk.evaluation._run_evaluation_inference",
            return_value=(["SELECT * FROM items"], ["SELECT * FROM items"], [0.95]),
        ),
    ):
        res = evaluate(
            "model",
            "dataset",
            "jax",
            ddl="CREATE TABLE items (id INT, val TEXT);",
            db_kwargs={"timeout": 5.0},
        )
        assert res["status"] == "completed"
        assert res["metrics"]["exact_match"] == pytest.approx(1.0)
        assert res["metrics"]["execution_accuracy"] == pytest.approx(1.0)


def test_evaluate_with_audio_path() -> None:
    """Verify evaluate preserves audio_path and modality in returned payload."""
    with (
        patch("gemma_4_sql.sdk.registry.get_backend"),
        patch(
            "gemma_4_sql.sdk.evaluation._run_evaluation_inference",
            return_value=(["SELECT 1"], ["SELECT 1"], [0.9]),
        ),
    ):
        res = evaluate("model", "dataset", "jax", audio_path="/path/to/query.wav", modality="audio")
        assert res["status"] == "completed"
        assert res.get("audio_path") == "/path/to/query.wav"
        assert res.get("modality") == "audio"


def test_evaluate_with_image_path() -> None:
    """Verify evaluate preserves image_path and modality in returned payload."""
    with (
        patch("gemma_4_sql.sdk.registry.get_backend"),
        patch(
            "gemma_4_sql.sdk.evaluation._run_evaluation_inference",
            return_value=(["SELECT 1"], ["SELECT 1"], [0.9]),
        ),
    ):
        res = evaluate("model", "dataset", "jax", image_path="/path/to/schema.png", modality="vision")
        assert res["status"] == "completed"
        assert res.get("image_path") == "/path/to/schema.png"
        assert res.get("modality") == "vision"


def test_collect_predictions_batch_multimodal() -> None:
    """Verify _run_evaluation_inference forwards pixel_values and audio_values from batches."""
    backend_impl = MagicMock()

    def mock_generate_sql(model_name: str, prompt: str, **_kwargs: JSONValue) -> InferenceResultDict:
        """Return simulated SQL and confidence score."""
        return {
            "backend": "jax",
            "model": model_name,
            "prompt": prompt,
            "sql": "SELECT 1",
            "beam_width": 3,
            "confidence_score": 0.85,
            "status": "success",
        }

    backend_impl.generate_sql.side_effect = mock_generate_sql

    multimodal_batch = {
        "inputs": [[1, 2]],
        "targets": [[2, 3]],
        "pixel_values": [[0.1] * 10],
        "audio_values": [[0.2] * 10],
    }
    backend_impl.build_dataloader.return_value = {"loader": [multimodal_batch]}

    preds, _truths, scores = _run_evaluation_inference(
        model_name="test-model",
        dataset_name="test-data",
        backend_impl=backend_impl,
        image_path="/path/to/image.png",
        audio_path="/path/to/audio.wav",
        modality="multimodal",
    )
    assert preds == ["SELECT 1"]
    assert scores == [0.85]

    preds2, _truths2, _scores2 = _run_evaluation_inference(
        model_name="test-model",
        dataset_name="test-data",
        backend_impl=backend_impl,
        modality="multimodal",
    )
    assert preds2 == ["SELECT 1"]


def test_evaluate_syntax_variations() -> None:
    """Verify normalize_sql handles case, comma spacing, operator spacing, parenthesis spacing, and comments."""
    sql1 = """  SELECT   id,name  FROM users WHERE ( id=1 AND age>=18 AND role!='admin' ) -- trailing comment
;  """
    sql2 = "select id, name from users where (id = 1 and age >= 18 and role != 'admin');"
    assert normalize_sql(sql1) == normalize_sql(sql2)
    assert normalize_sql("SELECT 'A=B' FROM t WHERE x<>2") == "select 'A=B' from t where x <> 2"


def test_evaluate_empty_metrics(synthetic_dataset: list[dict[str, str]]) -> None:
    """Verify compute_metrics returns 0.0 metrics for empty input lists and synthetic_dataset has expected items."""
    assert len(synthetic_dataset) == 2
    mock_engine = MagicMock()
    metrics = compute_metrics(mock_engine, [], [])
    assert metrics.get("exact_match") == 0.0
    assert metrics.get("execution_accuracy") == 0.0


def test_extract_batch_ids_dict_edge_cases() -> None:
    """Verify _process_batch_inputs handles dicts with missing keys, non-iterable values, and 1D sequences."""
    untyped_proc: Callable[..., tuple[list[int], list[int]]] = _process_batch_inputs
    inputs, targets = untyped_proc({"foo": "bar"})  # ty: ignore[invalid-argument-type]
    assert inputs == []
    assert targets == []

    inputs2, targets2 = untyped_proc({"inputs": 123, "targets": 456})  # ty: ignore[invalid-argument-type]
    assert inputs2 == []
    assert targets2 == []

    inputs3, targets3 = untyped_proc({"inputs": [1, 2, 3], "targets": [4, 5, 6]})  # ty: ignore[invalid-argument-type]
    assert inputs3 == [1, 2, 3]
    assert targets3 == [4, 5, 6]

    class ScalarTensor:
        """Stub returning a scalar int from .tolist()."""

        def tolist(self) -> int:
            """Return scalar int."""
            return 7

    inputs4, targets4 = untyped_proc(([ScalarTensor()], [ScalarTensor()]))  # ty: ignore[invalid-argument-type]
    assert inputs4 == [7]
    assert targets4 == [7]

    inputs5, targets5 = untyped_proc(([], []))
    assert inputs5 == []
    assert targets5 == []


@pytest.mark.asyncio
async def test_compute_metrics_async_mismatch_and_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify compute_metrics_async raises ValueError on length mismatch and handles execution exceptions."""
    engine = LiveDatabaseEngine(":memory:")
    try:
        with pytest.raises(ValueError, match="Length mismatch"):
            await compute_metrics_async(engine, ["a", "b"], ["a"])

        async def mock_exec(_query: str) -> tuple[bool, list[tuple[JSONPrimitive, ...]], str | None]:
            """Raise RuntimeError during execution."""
            raise RuntimeError("Mock error")

        monkeypatch.setattr(engine, "execute_with_feedback_async", mock_exec)
        metrics = await compute_metrics_async(engine, ["a"], ["a"])
        assert metrics["valid_sql"] == 0.0
        assert metrics["execution_accuracy"] == 0.0
    finally:
        engine.close()

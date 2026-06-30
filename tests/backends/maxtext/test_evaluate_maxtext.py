"""Tests for MaxText-specific model evaluation."""

import typing

import pytest

from gemma_4_sql.backends.maxtext.evaluate import evaluate_model


def test_evaluate_model_maxtext(monkeypatch: pytest.MonkeyPatch) -> None:
    """Function docstring."""
    monkeypatch.setattr("gemma_4_sql.backends.maxtext.evaluate.generate_sql", lambda *_a, **_k: {"sql": "SELECT 1"})
    """Test MaxText evaluate_model returns expected format."""
    res = evaluate_model("model1", "data1")
    if not res["backend"] == "maxtext":
        raise AssertionError
    if not res["model"] == "model1":
        raise AssertionError
    if not res["dataset"] == "data1":
        raise AssertionError
    if not res["metrics"]["execution_accuracy"] == 0.0:  # type: ignore[index]
        raise AssertionError


def test_evaluate_model_maxtext_mismatch() -> None:
    """Test MaxText evaluate_model with mismatching queries."""
    res = evaluate_model("model1", "data1", mock_predictions=["SELECT 1", "SELECT 2", "SELECT * FROM invalid"], mock_truths=["SELECT 1", "SELECT 3", "SELECT 4"])
    if not res["metrics"]["execution_accuracy"] == 1 / 3:  # type: ignore[index]
        raise AssertionError


def test_evaluate_model_maxtext_empty() -> None:
    """Test MaxText evaluate_model with empty queries."""
    res = evaluate_model("model1", "data1", mock_predictions=[], mock_truths=[])
    if not res["metrics"]["execution_accuracy"] == 0.0:  # type: ignore[index]
        raise AssertionError


def test_evaluate_model_with_dataloader(monkeypatch: object) -> None:
    """Test evaluate_model with a mocked dataloader."""
    importlib = __import__("importlib")
    sys = __import__("sys")

    def mock_build_dataloader(*_args: object, **_kwargs: object) -> object:
        """Initialize function mock_build_dataloader.

        Args:
        ----
        args: Description of args.
        kwargs: Description of kwargs.

        """

        class MockLoader:
            """Initialize class MockLoader."""

            def __iter__(self: typing.Any) -> object:
                """Initialize function __iter__."""
                yield {"inputs": [[101, 102]], "targets": [[101, 103]]}

        return {"loader": MockLoader(), "status": "loaded"}

    backend = ["tests/backends/maxtext", "test_evaluate_maxtext.py"][-1].replace("test_evaluate_", "").replace(".py", "")
    module_name = f"gemma_4_sql.backends.{backend}_approach.evaluate" if backend != "maxtext" else "gemma_4_sql.backends.maxtext.evaluate"
    importlib.import_module(module_name)
    monkeypatch.setattr(module_name + ".build_dataloader", mock_build_dataloader)  # type: ignore[attr-defined]
    eval_fn = sys.modules[module_name].evaluate_model
    monkeypatch.setattr(module_name + ".generate_sql", lambda *_a, **_k: {"sql": "SELECT 1"})
    res = eval_fn("model", "data")
    if "metrics" not in res:
        raise AssertionError


def test_evaluate_model_maxtext_fallback(monkeypatch: object) -> None:
    """Initialize function test_evaluate_model_maxtext_fallback.

    Args:
    ----
    monkeypatch: Description of monkeypatch.

    """
    evaluate = __import__("gemma_4_sql.backends.maxtext", fromlist=["evaluate"]).evaluate
    monkeypatch.setattr(evaluate, "build_dataloader", lambda *_args, **_kwargs: {})  # type: ignore[attr-defined]
    monkeypatch.setattr(evaluate, "generate_sql", lambda *_a, **_k: {"sql": "SELECT 1"})
    evaluate.evaluate_model("model1", "data1")

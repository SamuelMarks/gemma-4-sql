"""Tests for PyTorch-specific model evaluation."""

import typing

import pytest

from gemma_4_sql.backends.pytorch.evaluate import evaluate_model


def test_evaluate_model_pytorch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Function docstring."""
    monkeypatch.setattr("gemma_4_sql.backends.pytorch.evaluate.generate_sql", lambda *_a, **_k: {"sql": "SELECT 1"})
    """Test PyTorch evaluate_model returns expected format."""
    res = evaluate_model("model1", "data1")
    if not res["backend"] == "pytorch":
        raise AssertionError
    if not res["model"] == "model1":
        raise AssertionError
    if not res["dataset"] == "data1":
        raise AssertionError
    if not res["metrics"]["execution_accuracy"] == 1.0:  # type: ignore[index]
        raise AssertionError


def test_evaluate_model_pytorch_mismatch() -> None:
    """Test PyTorch evaluate_model with mismatching queries."""
    res = evaluate_model("model1", "data1", mock_predictions=["SELECT 1", "SELECT 2", "SELECT * FROM invalid"], mock_truths=["SELECT 1", "SELECT 3", "SELECT 4"])
    if not res["metrics"]["execution_accuracy"] == 1 / 3:  # type: ignore[index]
        raise AssertionError


def test_evaluate_model_pytorch_empty() -> None:
    """Test PyTorch evaluate_model with empty queries."""
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
                for _ in range(11):
                    yield {"inputs": [[101, 102]], "targets": [[101, 103]]}

        return {"loader": MockLoader(), "status": "loaded"}

    backend = ["tests/backends/pytorch", "test_evaluate_pytorch.py"][-1].replace("test_evaluate_", "").replace(".py", "")
    module_name = f"gemma_4_sql.backends.{backend}.evaluate"
    importlib.import_module(module_name)
    monkeypatch.setattr(module_name + ".build_dataloader", mock_build_dataloader)  # type: ignore[attr-defined]
    eval_fn = sys.modules[module_name].evaluate_model
    monkeypatch.setattr(module_name + ".generate_sql", lambda *_a, **_k: {"sql": "SELECT 1"})
    res = eval_fn("model", "data")
    if "metrics" not in res:
        raise AssertionError

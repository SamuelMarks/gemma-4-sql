"""
Tests for PyTorch-specific model evaluation.
"""

from gemma_4_sql.backends.pytorch.evaluate import evaluate_model


def test_evaluate_model_pytorch() -> None:
    """Test PyTorch evaluate_model returns expected format."""
    res = evaluate_model("model1", "data1")
    assert res["backend"] == "pytorch"
    assert res["model"] == "model1"
    assert res["dataset"] == "data1"
    assert res["metrics"]["execution_accuracy"] == 1.0  # Empty tables return [] == []
    assert res["metrics"]["valid_sql"] == 0.0


def test_evaluate_model_pytorch_mismatch() -> None:
    """Test PyTorch evaluate_model with mismatching queries."""
    res = evaluate_model(
        "model1",
        "data1",
        mock_predictions=["SELECT 1", "SELECT 2", "SELECT * FROM invalid"],
        mock_truths=["SELECT 1", "SELECT 3", "SELECT 4"],
    )
    assert res["metrics"]["execution_accuracy"] == 1 / 3
    assert res["metrics"]["exact_match"] == 1 / 3
    assert res["metrics"]["valid_sql"] == 2 / 3


def test_evaluate_model_pytorch_empty() -> None:
    """Test PyTorch evaluate_model with empty queries."""
    res = evaluate_model("model1", "data1", mock_predictions=[], mock_truths=[])
    assert res["metrics"]["execution_accuracy"] == 0.0
    assert res["metrics"]["exact_match"] == 0.0
    assert res["metrics"]["valid_sql"] == 0.0


def test_evaluate_model_with_dataloader(monkeypatch) -> None:
    """Test evaluate_model with a mocked dataloader."""
    import importlib
    import sys

    def mock_build_dataloader(*args, **kwargs):
        class MockLoader:
            def __iter__(self):
                # Yield 11 batches to trigger break condition
                for _ in range(11):
                    yield {"inputs": [[101, 102]], "targets": [[101, 103]]}

        return {"loader": MockLoader(), "status": "loaded"}

    # Get backend from file name
    backend = (
        "tests/backends/pytorch/test_evaluate_pytorch.py".split("/")[-1]
        .replace("test_evaluate_", "")
        .replace(".py", "")
    )
    module_name = f"gemma_4_sql.backends.{backend}.evaluate"

    importlib.import_module(module_name)
    monkeypatch.setattr(module_name + ".build_dataloader", mock_build_dataloader)

    eval_fn = sys.modules[module_name].evaluate_model
    res = eval_fn("model", "data")
    assert res["status"] == "completed"
    assert "metrics" in res

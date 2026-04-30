"""
Tests for Keras-specific model evaluation.
"""

from gemma_4_sql.backends.keras.evaluate import evaluate_model


def test_evaluate_model_keras() -> None:
    """Test Keras evaluate_model returns expected format."""
    res = evaluate_model("model1", "data1")
    assert res["backend"] == "keras"
    assert res["model"] == "model1"
    assert res["dataset"] == "data1"
    assert res["metrics"]["execution_accuracy"] == 1.0  # Empty tables return [] == []
    assert res["metrics"]["valid_sql"] == 0.0


def test_evaluate_model_keras_mismatch() -> None:
    """Test Keras evaluate_model with mismatching queries."""
    res = evaluate_model(
        "model1",
        "data1",
        mock_predictions=["SELECT 1", "SELECT 2", "SELECT * FROM invalid"],
        mock_truths=["SELECT 1", "SELECT 3", "SELECT 4"],
    )
    assert res["metrics"]["execution_accuracy"] == 1 / 3
    assert res["metrics"]["exact_match"] == 1 / 3
    assert res["metrics"]["valid_sql"] == 2 / 3


def test_evaluate_model_keras_empty() -> None:
    """Test Keras evaluate_model with empty queries."""
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
                # Yield a single batch
                yield {"inputs": [[101, 102]], "targets": [[101, 103]]}

        return {"loader": MockLoader(), "status": "loaded"}

    # Get backend from file name
    backend = (
        "tests/backends/keras/test_evaluate_keras.py".split("/")[-1]
        .replace("test_evaluate_", "")
        .replace(".py", "")
    )
    module_name = (
        f"gemma_4_sql.backends.{backend}.evaluate"
        if backend != "maxtext"
        else "gemma_4_sql.backends.maxtext.evaluate"
    )

    importlib.import_module(module_name)
    monkeypatch.setattr(module_name + ".build_dataloader", mock_build_dataloader)

    eval_fn = sys.modules[module_name].evaluate_model
    res = eval_fn("model", "data")
    assert res["status"] == "completed"
    assert "metrics" in res


def test_evaluate_model_keras_fallback(monkeypatch) -> None:
    from gemma_4_sql.backends.keras import evaluate

    monkeypatch.setattr(evaluate, "build_dataloader", lambda *args, **kwargs: {})
    res = evaluate.evaluate_model("model1", "data1")
    assert res["status"] == "completed"

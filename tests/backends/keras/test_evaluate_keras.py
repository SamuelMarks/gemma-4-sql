"""Tests for Keras-specific model evaluation."""

import typing

import pytest

from gemma_4_sql.backends.keras.evaluate import evaluate_model


def test_evaluate_model_keras(monkeypatch: pytest.MonkeyPatch) -> None:
    """Function docstring."""
    monkeypatch.setattr("gemma_4_sql.backends.keras.evaluate.generate_sql", lambda *_a, **_k: {"sql": "SELECT 1"})
    """Test Keras evaluate_model returns expected format."""
    res = evaluate_model("model1", "data1")
    if not res["backend"] == "keras":
        raise AssertionError
    if not res["model"] == "model1":
        raise AssertionError
    if not res["dataset"] == "data1":
        raise AssertionError
    if not res["metrics"]["execution_accuracy"] == 0.0:  # type: ignore[index]
        raise AssertionError


def test_evaluate_model_keras_mismatch() -> None:
    """Test Keras evaluate_model with mismatching queries."""
    res = evaluate_model("model1", "data1", mock_predictions=["SELECT 1", "SELECT 2", "SELECT * FROM invalid"], mock_truths=["SELECT 1", "SELECT 3", "SELECT 4"])
    if not res["metrics"]["execution_accuracy"] == 1 / 3:  # type: ignore[index]
        raise AssertionError


def test_evaluate_model_keras_empty() -> None:
    """Test Keras evaluate_model with empty queries."""
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

    backend = ["tests/backends/keras", "test_evaluate_keras.py"][-1].replace("test_evaluate_", "").replace(".py", "")
    module_name = f"gemma_4_sql.backends.{backend}.evaluate" if backend != "maxtext" else "gemma_4_sql.backends.maxtext.evaluate"
    importlib.import_module(module_name)
    monkeypatch.setattr(module_name + ".build_dataloader", mock_build_dataloader)  # type: ignore[attr-defined]
    eval_fn = sys.modules[module_name].evaluate_model
    monkeypatch.setattr(module_name + ".generate_sql", lambda *_a, **_k: {"sql": "SELECT 1"})
    res = eval_fn("model", "data")
    if "metrics" not in res:
        raise AssertionError


def test_evaluate_model_keras_fallback(monkeypatch: object) -> None:
    """Initialize function test_evaluate_model_keras_fallback.

    Args:
    ----
    monkeypatch: Description of monkeypatch.

    """
    evaluate = __import__("gemma_4_sql.backends.keras", fromlist=["evaluate"]).evaluate
    monkeypatch.setattr(evaluate, "build_dataloader", lambda *_args, **_kwargs: {})  # type: ignore[attr-defined]
    monkeypatch.setattr(evaluate, "generate_sql", lambda *_a, **_k: {"sql": "SELECT 1"})
    evaluate.evaluate_model("model1", "data1")


def test_evaluate_model_with_dataloader_tuple(monkeypatch: pytest.MonkeyPatch) -> None:
    import gemma_4_sql.backends.keras.evaluate as ke

    class MockKeras:
        class Model:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

        Input = lambda *args, **kwargs: None

        class layers:
            Embedding = lambda *args, **kwargs: lambda x: "x"
            Dense = lambda *args, **kwargs: lambda x: "x"

    class MockTfTensor:
        def __init__(self, data):
            self.data = data

        def numpy(self):
            data = self.data

            class MockNumpy:
                def tolist(self):
                    return data

            return MockNumpy()

        def __getitem__(self, idx):
            return self.data[idx]

        def __len__(self):
            return len(self.data)

    class MockTokenizer:
        def __init__(self, *args, **kwargs):
            pass

        def decode(self, *args, **kwargs):
            return "SELECT 1"

    def mock_build_dataloader(*args: object, **kwargs: object) -> dict:
        return {"loader": [([MockTfTensor([1, 2, 3])], [MockTfTensor([4, 5, 6])])]}

    monkeypatch.setattr(ke, "build_dataloader", mock_build_dataloader)
    monkeypatch.setattr(ke, "generate_sql", lambda *args, **kwargs: {"sql": "SELECT 1"})
    monkeypatch.setattr(ke, "SQLTokenizer", MockTokenizer)
    res = ke.evaluate_model("model", "ds", db_path=":memory:")
    assert res["status"] == "completed"

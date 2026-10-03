import sqlite3
from typing import ClassVar

import pytest

from gemma_4_sql.backends.keras.train import train_model
from gemma_4_sql.type_hints import TrainingConfig

"""Real Keras train tests."""


def test_train_model_keras(tmp_path) -> None:
    # Use real mock sqlite DB for dataloader
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE tbl (question VARCHAR, query VARCHAR)")
    conn.execute("INSERT INTO tbl VALUES ('What is 1?', 'SELECT 1')")
    conn.close()

    config = TrainingConfig(
        action="sft",
        model_name="dummy",
        dataset="dummy",
        epochs=1,
        batch_size=1,
        extra_kwargs={"duckdb_path": str(db_path), "duckdb_table": "tbl"},
    )
    # The actual train_model function will raise ValueError because dummy is not a real Keras preset
    # but we just want to ensure it executes the pipeline until that point without mock errors.
    res = train_model(config)
    assert "failed" in res["status"]


def test_execute_train_success(monkeypatch: pytest.MonkeyPatch) -> None:
    class MockHistory:
        history: ClassVar[dict] = {"loss": (0.5,)}

    class MockModel:
        preprocessor = type("MockPrep", (), {"sequence_length": 512})()

        def compile(self, *a, **k):
            pass

        def fit(self, *a, **k):
            return MockHistory()

    import gemma_4_sql.backends.keras.train as tr

    monkeypatch.setattr(tr, "build_dataloader", lambda *a, **k: {"loader": [{"a": 1}]})

    # We have to mock tf strategy and keras_nlp
    class MockScope:
        def __enter__(self):
            pass

        def __exit__(self, *a):
            pass

    class MockStrategy:
        def scope(self):
            return MockScope()

    monkeypatch.setattr(tr.tf.distribute, "MirroredStrategy", MockStrategy)

    # Mock keras_nlp
    import sys

    class MockGemmaCls:
        @classmethod
        def from_preset(cls, *a):
            return MockModel()

    mock_keras_nlp = type("keras_nlp", (), {"models": type("models", (), {"GemmaCausalLM": MockGemmaCls})})
    monkeypatch.setitem(sys.modules, "keras_nlp.models", mock_keras_nlp.models)

    status, loss = tr._execute_train("mod", "ds", 1, 2)
    assert status == "completed"
    assert loss == 0.5

    # Also test train_model wrapper wrapper success
    res = tr.train_model(TrainingConfig(action="sft", model_name="mod", dataset="ds", epochs=1, batch_size=2))
    assert res["status"] == "completed"
    assert res["final_loss"] == 0.5


def test_execute_train_no_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    class MockHistory:
        history: tuple = ()

    class MockModel:
        preprocessor = type("MockPrep", (), {"sequence_length": 512})()

        def compile(self, *a, **k):
            pass

        def fit(self, *a, **k):
            return MockHistory()

    import gemma_4_sql.backends.keras.train as tr

    monkeypatch.setattr(tr, "build_dataloader", lambda *a, **k: {"loader": [{"a": 1}]})

    class MockScope:
        def __enter__(self):
            pass

        def __exit__(self, *a):
            pass

    class MockStrategy:
        def scope(self):
            return MockScope()

    monkeypatch.setattr(tr.tf.distribute, "MirroredStrategy", MockStrategy)

    import sys

    class MockGemmaCls:
        @classmethod
        def from_preset(cls, *a):
            return MockModel()

    mock_keras_nlp = type("keras_nlp", (), {"models": type("models", (), {"GemmaCausalLM": MockGemmaCls})})
    monkeypatch.setitem(sys.modules, "keras_nlp.models", mock_keras_nlp.models)

    status, loss = tr._execute_train("mod", "ds", 1, 2)
    assert status == "completed"
    assert loss == 0.0


def test_execute_train_load_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    import gemma_4_sql.backends.keras.train as tr

    class MockScope:
        def __enter__(self):
            pass

        def __exit__(self, *a):
            pass

    class MockStrategy:
        def scope(self):
            return MockScope()

    monkeypatch.setattr(tr.tf.distribute, "MirroredStrategy", MockStrategy)

    import sys

    class MockGemmaCls:
        @classmethod
        def from_preset(cls, *a):
            raise ValueError("Cannot load")

    mock_keras_nlp = type("keras_nlp", (), {"models": type("models", (), {"GemmaCausalLM": MockGemmaCls})})
    monkeypatch.setitem(sys.modules, "keras_nlp.models", mock_keras_nlp.models)

    with pytest.raises(ValueError, match="Failed to load Keras model"):
        tr._execute_train("mod", "ds", 1, 2)


def test_execute_train_bad_dataloader(monkeypatch: pytest.MonkeyPatch) -> None:
    import gemma_4_sql.backends.keras.train as tr

    monkeypatch.setattr(tr, "build_dataloader", lambda *a, **k: {"loader": None})

    class MockScope:
        def __enter__(self):
            pass

        def __exit__(self, *a):
            pass

    class MockStrategy:
        def scope(self):
            return MockScope()

    monkeypatch.setattr(tr.tf.distribute, "MirroredStrategy", MockStrategy)

    import sys

    class MockGemmaCls:
        @classmethod
        def from_preset(cls, *a):
            return type("MockModel", (), {"preprocessor": type("MockPrep", (), {"sequence_length": 512})(), "compile": lambda *args, **kwargs: None})()

    mock_keras_nlp = type("keras_nlp", (), {"models": type("models", (), {"GemmaCausalLM": MockGemmaCls})})
    monkeypatch.setitem(sys.modules, "keras_nlp.models", mock_keras_nlp.models)

    with pytest.raises(ValueError, match="Invalid dataloader"):
        tr._execute_train("mod", "ds", 1, 2)


def test_train_model_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    import gemma_4_sql.backends.keras.train as tr
    from gemma_4_sql.exceptions import DependencyMissingError

    monkeypatch.setattr(tr, "keras", None)
    with pytest.raises(DependencyMissingError):
        tr.train_model(TrainingConfig(model_name="m", dataset="d"))
    with pytest.raises(DependencyMissingError):
        tr._execute_train("m", "d", 1, 1)

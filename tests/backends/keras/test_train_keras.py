import sqlite3
from typing import ClassVar

import pytest

from gemma_4_sql.type_hints import TrainingConfig


@pytest.fixture
def mock_keras_env(monkeypatch: pytest.MonkeyPatch):
    import gemma_4_sql.backends.keras.train as tr

    class MockKeras:
        class losses:
            @staticmethod
            def SparseCategoricalCrossentropy(*args, **kwargs):
                return "loss"

        class optimizers:
            @staticmethod
            def AdamW(*args, **kwargs):
                return "optimizer"

    class MockTF:
        class distribute:
            class MirroredStrategy:
                def scope(self):
                    class MockScope:
                        def __enter__(self):
                            pass

                        def __exit__(self, *a):
                            pass

                    return MockScope()

    monkeypatch.setattr(tr, "keras", MockKeras)
    monkeypatch.setattr(tr, "tf", MockTF)
    return tr


def test_train_model_keras(tmp_path, mock_keras_env, monkeypatch: pytest.MonkeyPatch) -> None:
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

    # Mock keras_nlp so it raises ValueError
    import sys

    class MockGemmaCls:
        @classmethod
        def from_preset(cls, *a):
            raise ValueError("dummy")

    mock_keras_nlp = type("keras_nlp", (), {"models": type("models", (), {"GemmaCausalLM": MockGemmaCls})})
    monkeypatch.setitem(sys.modules, "keras_nlp.models", mock_keras_nlp.models)

    res = mock_keras_env.train_model(config)
    assert "failed" in res["status"]


def test_execute_train_success(mock_keras_env, monkeypatch: pytest.MonkeyPatch) -> None:
    class MockHistory:
        history: ClassVar[dict] = {"loss": (0.5,)}

    class MockModel:
        preprocessor = type("MockPrep", (), {"sequence_length": 512})()

        def compile(self, *a, **k):
            pass

        def fit(self, *a, **k):
            return MockHistory()

    monkeypatch.setattr(mock_keras_env, "build_dataloader", lambda *a, **k: {"loader": [{"a": 1}]})

    import sys

    class MockGemmaCls:
        @classmethod
        def from_preset(cls, *a):
            return MockModel()

    mock_keras_nlp = type("keras_nlp", (), {"models": type("models", (), {"GemmaCausalLM": MockGemmaCls})})
    monkeypatch.setitem(sys.modules, "keras_nlp.models", mock_keras_nlp.models)

    status, loss = mock_keras_env._execute_train("mod", "ds", 1, 2)
    assert status == "completed"
    assert loss == 0.5

    res = mock_keras_env.train_model(TrainingConfig(action="sft", model_name="mod", dataset="ds", epochs=1, batch_size=2))
    assert res["status"] == "completed"
    assert res["final_loss"] == 0.5


def test_execute_train_no_loss(mock_keras_env, monkeypatch: pytest.MonkeyPatch) -> None:
    class MockHistory:
        history: tuple = ()

    class MockModel:
        preprocessor = type("MockPrep", (), {"sequence_length": 512})()

        def compile(self, *a, **k):
            pass

        def fit(self, *a, **k):
            return MockHistory()

    monkeypatch.setattr(mock_keras_env, "build_dataloader", lambda *a, **k: {"loader": [{"a": 1}]})

    import sys

    class MockGemmaCls:
        @classmethod
        def from_preset(cls, *a):
            return MockModel()

    mock_keras_nlp = type("keras_nlp", (), {"models": type("models", (), {"GemmaCausalLM": MockGemmaCls})})
    monkeypatch.setitem(sys.modules, "keras_nlp.models", mock_keras_nlp.models)

    status, loss = mock_keras_env._execute_train("mod", "ds", 1, 2)
    assert status == "completed"
    assert loss == 0.0


def test_execute_train_load_fail(mock_keras_env, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    class MockGemmaCls:
        @classmethod
        def from_preset(cls, *a):
            raise ValueError("Cannot load")

    mock_keras_nlp = type("keras_nlp", (), {"models": type("models", (), {"GemmaCausalLM": MockGemmaCls})})
    monkeypatch.setitem(sys.modules, "keras_nlp.models", mock_keras_nlp.models)

    with pytest.raises(ValueError, match="Failed to load Keras model"):
        mock_keras_env._execute_train("mod", "ds", 1, 2)


def test_execute_train_bad_dataloader(mock_keras_env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mock_keras_env, "build_dataloader", lambda *a, **k: {"loader": None})

    import sys

    class MockGemmaCls:
        @classmethod
        def from_preset(cls, *a):
            return type("MockModel", (), {"preprocessor": type("MockPrep", (), {"sequence_length": 512})(), "compile": lambda *args, **kwargs: None})()

    mock_keras_nlp = type("keras_nlp", (), {"models": type("models", (), {"GemmaCausalLM": MockGemmaCls})})
    monkeypatch.setitem(sys.modules, "keras_nlp.models", mock_keras_nlp.models)

    with pytest.raises(ValueError, match="Invalid dataloader"):
        mock_keras_env._execute_train("mod", "ds", 1, 2)


def test_train_model_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    import gemma_4_sql.backends.keras.train as tr
    from gemma_4_sql.exceptions import DependencyMissingError

    monkeypatch.setattr(tr, "keras", None)
    with pytest.raises(DependencyMissingError, match="Keras training dependencies are missing."):
        tr.train_model(TrainingConfig(model_name="m", dataset="d"))
    with pytest.raises(DependencyMissingError, match="Keras dependencies are missing."):
        tr._execute_train("m", "d", 1, 1)

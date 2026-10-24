"""Unit and behavioral tests for Keras training pipeline."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Any

import pytest

from gemma_4_sql.exceptions import DependencyMissingError, TrainingError
from gemma_4_sql.type_hints import TrainingConfig
from tests.conftest import requires_keras

pytestmark = requires_keras


def _make_tiny_keras_model() -> Any:
    """Create a real tiny Keras functional model with a preprocessor attribute."""
    import keras

    inputs = keras.Input(shape=(4,), dtype="int32")
    x = keras.layers.Embedding(input_dim=32, output_dim=8)(inputs)
    outputs = keras.layers.Dense(32)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.preprocessor = type("Preprocessor", (), {"sequence_length": 512})()  # type: ignore[attr-defined]
    return model


@pytest.fixture
def real_keras_env(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Configure keras.train with real Keras and a lightweight MirroredStrategy scope."""
    import keras

    import gemma_4_sql.backends.keras.train as tr

    class _Scope:
        """No-op strategy scope context manager."""

        def __enter__(self) -> None:
            """Enter scope."""

        def __exit__(self, *_a: object) -> None:
            """Exit scope."""

    class _MirroredStrategy:
        """Lightweight MirroredStrategy stub avoiding multi-GPU device probing on CPU."""

        def scope(self) -> _Scope:
            """Return no-op context manager."""
            return _Scope()

    class _DistributeNamespace:
        """Namespace providing MirroredStrategy."""

        MirroredStrategy = _MirroredStrategy

    class _TFNamespace:
        """Minimal TensorFlow distribute namespace."""

        distribute = _DistributeNamespace

    monkeypatch.setattr(tr, "keras", keras)
    monkeypatch.setattr(tr, "tf", _TFNamespace)
    return tr


def test_train_model_keras(tmp_path: Path, real_keras_env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify train_model handles preset loading errors and propagates TrainingError when requested."""
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

    class FailingGemmaCls:
        """Stub GemmaCausalLM class that raises ValueError on from_preset."""

        @classmethod
        def from_preset(cls, *_a: object) -> Any:
            """Raise ValueError."""
            raise ValueError("dummy")

    monkeypatch.setattr(real_keras_env, "GemmaCausalLM", FailingGemmaCls)

    res = real_keras_env.train_model(config)
    assert "failed" in res["status"]

    with pytest.raises(TrainingError, match="Failed to load Keras model"):
        real_keras_env.train_model(config, raise_on_error=True)


def test_execute_train_success_real_keras_model(real_keras_env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _execute_train and train_model compile and fit a real tiny keras.Model with real SparseCategoricalCrossentropy and AdamW."""
    import numpy as np

    tiny_model = _make_tiny_keras_model()
    w_before = np.array(tiny_model.layers[-1].get_weights()[0], copy=True)

    x_batch = np.array([[1, 2, 3, 4], [2, 3, 4, 5]], dtype="int32")
    y_batch = np.array([[2, 3, 4, 5], [3, 4, 5, 6]], dtype="int32")

    def _batch_gen() -> Any:
        """Yield a single `(x_batch, y_batch)` tuple for `keras.Model.fit`."""
        yield (x_batch, y_batch)

    monkeypatch.setattr(real_keras_env, "build_dataloader", lambda *_a, **_k: {"loader": _batch_gen()})

    class RealGemmaCls:
        """GemmaCausalLM provider returning the real tiny keras.Model."""

        @classmethod
        def from_preset(cls, *_a: object) -> Any:
            """Return tiny_model."""
            return tiny_model

    monkeypatch.setattr(real_keras_env, "GemmaCausalLM", RealGemmaCls)

    status, loss = real_keras_env._execute_train("mod", "ds", 1, 2)
    assert status == "completed"
    assert loss > 0.0
    w_after = np.array(tiny_model.layers[-1].get_weights()[0], copy=True)
    assert not np.allclose(w_before, w_after)

    res = real_keras_env.train_model(TrainingConfig(action="sft", model_name="mod", dataset="ds", epochs=1, batch_size=2))
    assert res["status"] == "completed"
    assert res["final_loss"] > 0.0


def test_execute_train_no_loss(real_keras_env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _execute_train defaults final_loss to 0.0 when history.history lacks 'loss'."""
    tiny_model = _make_tiny_keras_model()
    monkeypatch.setattr(tiny_model, "fit", lambda *_a, **_k: type("EmptyHistory", (), {"history": {}})())
    monkeypatch.setattr(real_keras_env, "build_dataloader", lambda *_a, **_k: {"loader": [(1, 2)]})
    monkeypatch.setattr(
        real_keras_env,
        "GemmaCausalLM",
        type("GemmaCls", (), {"from_preset": classmethod(lambda _cls, *_a: tiny_model)}),
    )

    status, loss = real_keras_env._execute_train("mod", "ds", 1, 2)
    assert status == "completed"
    assert loss == 0.0


def test_execute_train_load_fail(real_keras_env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _execute_train raises ValueError when GemmaCausalLM cannot be loaded or imported."""

    class FailingGemmaCls:
        """Stub raising ValueError on from_preset."""

        @classmethod
        def from_preset(cls, *_a: object) -> Any:
            """Raise ValueError."""
            raise ValueError("Cannot load")

    monkeypatch.setattr(real_keras_env, "GemmaCausalLM", FailingGemmaCls)
    with pytest.raises(ValueError, match="Failed to load Keras model"):
        real_keras_env._execute_train("mod", "ds", 1, 2)

    monkeypatch.setattr(real_keras_env, "GemmaCausalLM", None)
    monkeypatch.setattr(real_keras_env, "optional_attr", lambda *_a, **_k: None)
    with pytest.raises(ValueError, match="Failed to load Keras model"):
        real_keras_env._execute_train("mod", "ds", 1, 2)


def test_execute_train_bad_dataloader(real_keras_env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _execute_train raises ValueError when build_dataloader returns a non-iterable loader."""
    tiny_model = _make_tiny_keras_model()
    monkeypatch.setattr(real_keras_env, "build_dataloader", lambda *_a, **_k: {"loader": None})
    monkeypatch.setattr(
        real_keras_env,
        "GemmaCausalLM",
        type("GemmaCls", (), {"from_preset": classmethod(lambda _cls, *_a: tiny_model)}),
    )
    monkeypatch.setitem(sys.modules, "keras_nlp", type("keras_nlp", (), {}))

    with pytest.raises(ValueError, match="Invalid dataloader"):
        real_keras_env._execute_train("mod", "ds", 1, 2)


def test_train_model_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify train_model and _execute_train raise DependencyMissingError when keras is None."""
    import gemma_4_sql.backends.keras.train as tr

    monkeypatch.setattr(tr, "keras", None)
    with pytest.raises(DependencyMissingError, match="Keras training dependencies are missing."):
        tr.train_model(TrainingConfig(model_name="m", dataset="d"))
    with pytest.raises(DependencyMissingError, match="Keras dependencies are missing."):
        tr._execute_train("m", "d", 1, 1)

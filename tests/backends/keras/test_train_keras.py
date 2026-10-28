"""Unit and behavioral tests for Keras training pipeline."""

from __future__ import annotations

import sqlite3
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import NoReturn

import keras
import pytest
from keras import ops

import gemma_4_sql.backends.keras.train as tr
from gemma_4_sql.exceptions import DependencyMissingError, TrainingError
from gemma_4_sql.type_hints import TensorProtocol, TrainingConfig
from tests.conftest import requires_keras

pytestmark = requires_keras

type IntBatch = dict[str, TensorProtocol] | tuple[TensorProtocol, TensorProtocol]


def _make_tiny_keras_model() -> keras.Model:
    """Create a real tiny Keras functional model with a preprocessor attribute."""
    inputs = keras.Input(shape=(4,), dtype="int32")
    x = keras.layers.Embedding(input_dim=32, output_dim=8)(inputs)
    outputs = keras.layers.Dense(32)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.preprocessor = type("Preprocessor", (), {"sequence_length": 512})()
    return model


@pytest.fixture
def real_keras_env(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Configure keras.train with real Keras and a lightweight MirroredStrategy scope."""

    class _Scope:
        """No-op strategy scope context manager."""

        def __enter__(self) -> None:
            """Enter scope."""

        def __exit__(
            self,
            _exc_type: type[BaseException] | None = None,
            _exc_val: BaseException | None = None,
            _exc_tb: types.TracebackType | None = None,
        ) -> None:
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
        """Minimal TensorFlow distribute and constant namespace."""

        distribute = _DistributeNamespace
        int32 = "int32"

        @staticmethod
        def constant(
            val: TensorProtocol | list[int] | int,
            dtype: str = "int32",
        ) -> TensorProtocol:
            """Convert array-like to Keras tensor with requested dtype."""
            return ops.convert_to_tensor(val, dtype=dtype)  # pyright: ignore[reportReturnType]

    monkeypatch.setattr(tr, "keras", keras)
    monkeypatch.setattr(tr, "tf", _TFNamespace)
    return tr


def test_train_model_keras(
    tmp_path: Path,
    real_keras_env: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify train_model handles preset loading errors and propagates TrainingError when requested."""
    _ = real_keras_env
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
        def from_preset(cls, _model_name: str) -> NoReturn:
            """Raise ValueError."""
            raise ValueError("dummy")

    monkeypatch.setattr(tr, "GemmaCausalLM", FailingGemmaCls)

    res = tr.train_model(config)
    assert "failed" in res["status"]

    with pytest.raises(TrainingError, match="Failed to load Keras model"):
        tr.train_model(config, raise_on_error=True)


def test_execute_train_success_real_keras_model(
    real_keras_env: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify _execute_train and train_model compile and fit a real tiny keras.Model with real SparseCategoricalCrossentropy and AdamW."""
    _ = real_keras_env
    tiny_model = _make_tiny_keras_model()
    w_before = ops.copy(ops.stop_gradient(tiny_model.layers[-1].weights[0]))

    x_batch: TensorProtocol = ops.convert_to_tensor([[1, 2, 3, 4], [2, 3, 4, 5]], dtype="int32")  # pyright: ignore[reportAssignmentType]
    y_batch: TensorProtocol = ops.convert_to_tensor([[2, 3, 4, 5], [3, 4, 5, 6]], dtype="int32")  # pyright: ignore[reportAssignmentType]
    yielded_batches = 0

    def _batch_gen() -> Iterator[IntBatch]:
        """Yield multiple dict and tuple batches for `keras.Model.fit`."""
        nonlocal yielded_batches
        yielded_batches += 1
        yield {"inputs": x_batch, "targets": y_batch}
        yielded_batches += 1
        yield (x_batch, y_batch)

    monkeypatch.setattr(tr, "build_dataloader", lambda *_a, **_k: {"loader": _batch_gen()})  # pyright: ignore[reportUnknownLambdaType]

    class RealGemmaCls:
        """GemmaCausalLM provider returning the real tiny keras.Model."""

        @classmethod
        def from_preset(cls, _model_name: str) -> keras.Model:
            """Return tiny_model."""
            return tiny_model

    monkeypatch.setattr(tr, "GemmaCausalLM", RealGemmaCls)

    status, loss = tr._execute_train("mod", "ds", 1, 2)
    assert status == "completed"
    assert loss > 0.0
    assert yielded_batches == 2
    w_after = ops.stop_gradient(tiny_model.layers[-1].weights[0])
    assert not bool(ops.all(ops.isclose(w_before, w_after)))

    res = tr.train_model(TrainingConfig(action="sft", model_name="mod", dataset="ds", epochs=1, batch_size=2))
    assert res["status"] == "completed"
    assert res["final_loss"] > 0.0


def test_execute_train_no_loss(
    real_keras_env: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify _execute_train defaults final_loss to 0.0 when history.history lacks 'loss'."""
    _ = real_keras_env
    tiny_model = _make_tiny_keras_model()
    monkeypatch.setattr(tiny_model, "fit", lambda *_a, **_k: type("EmptyHistory", (), {"history": {}})())  # pyright: ignore[reportUnknownLambdaType]
    monkeypatch.setattr(tr, "build_dataloader", lambda *_a, **_k: {"loader": [(1, 2)]})  # pyright: ignore[reportUnknownLambdaType]
    monkeypatch.setattr(
        tr,
        "GemmaCausalLM",
        type("GemmaCls", (), {"from_preset": classmethod(lambda _cls, *_a: tiny_model)}),
    )

    status, loss = tr._execute_train("mod", "ds", 1, 2)
    assert status == "completed"
    assert loss == 0.0


def test_execute_train_load_fail(
    real_keras_env: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify _execute_train raises ValueError when GemmaCausalLM cannot be loaded or imported."""
    _ = real_keras_env

    class FailingGemmaCls:
        """Stub raising ValueError on from_preset."""

        @classmethod
        def from_preset(cls, _model_name: str) -> NoReturn:
            """Raise ValueError."""
            raise ValueError("Cannot load")

    monkeypatch.setattr(tr, "GemmaCausalLM", FailingGemmaCls)
    with pytest.raises(ValueError, match="Failed to load Keras model"):
        tr._execute_train("mod", "ds", 1, 2)

    monkeypatch.setattr(tr, "GemmaCausalLM", None)
    monkeypatch.setattr(tr, "optional_attr", lambda *_a, **_k: None)  # pyright: ignore[reportUnknownLambdaType]
    with pytest.raises(ValueError, match="Failed to load Keras model"):
        tr._execute_train("mod", "ds", 1, 2)


def test_execute_train_bad_dataloader(
    real_keras_env: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify _execute_train raises ValueError when build_dataloader returns a non-iterable loader."""
    _ = real_keras_env
    tiny_model = _make_tiny_keras_model()
    monkeypatch.setattr(tr, "build_dataloader", lambda *_a, **_k: {"loader": None})  # pyright: ignore[reportUnknownLambdaType]
    monkeypatch.setattr(
        tr,
        "GemmaCausalLM",
        type("GemmaCls", (), {"from_preset": classmethod(lambda _cls, *_a: tiny_model)}),
    )
    monkeypatch.setitem(sys.modules, "keras_nlp", type("keras_nlp", (), {}))

    with pytest.raises(ValueError, match="Invalid dataloader"):
        tr._execute_train("mod", "ds", 1, 2)


def test_train_model_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify train_model and _execute_train raise DependencyMissingError when keras is None."""
    monkeypatch.setattr(tr, "keras", None)
    with pytest.raises(DependencyMissingError, match=r"Keras training dependencies are missing\."):
        tr.train_model(TrainingConfig(model_name="m", dataset="d"))
    with pytest.raises(DependencyMissingError, match=r"Keras dependencies are missing\."):
        tr._execute_train("m", "d", 1, 1)

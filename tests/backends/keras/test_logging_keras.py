"""Unit and behavioral tests for Keras metrics logging without untyped mocks."""

from __future__ import annotations

import types
from pathlib import Path
from typing import Self

import pytest

import gemma_4_sql.backends.keras.logging as log_keras
from gemma_4_sql.backends.keras.logging import log_metrics
from tests.conftest import requires_keras

pytestmark = requires_keras


def test_log_metrics_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify Keras `log_metrics` creates a TensorFlow summary writer and records scalar metrics."""
    created_dirs: list[str] = []
    recorded_scalars: list[tuple[str, float, int]] = []
    closed_count = 0

    class _FakeWriterCtx:
        """Context manager returned by `writer.as_default()`."""

        def __enter__(self) -> Self:
            """Enter default writer context.

            Returns:
                This `_FakeWriterCtx` instance.

            """
            return self

        def __exit__(
            self,
            _exc_type: type[BaseException] | None,
            _exc: BaseException | None,
            _tb: types.TracebackType | None,
        ) -> None:
            """Exit default writer context."""

    class _FakeFileWriter:
        """Typed TensorFlow summary file writer double."""

        def as_default(self) -> _FakeWriterCtx:
            """Return context manager."""
            return _FakeWriterCtx()

        def flush(self) -> None:
            """No-op flush."""

        def close(self) -> None:
            """Increment `closed_count`."""
            nonlocal closed_count
            closed_count += 1

    class _FakeSummaryNamespace:
        """Typed `tf.summary` double."""

        def create_file_writer(self, logdir: str) -> _FakeFileWriter:
            """Record `logdir` and return `_FakeFileWriter`.

            Returns:
                A `_FakeFileWriter` instance.

            """
            created_dirs.append(logdir)
            return _FakeFileWriter()

        def scalar(self, name: str, data: float, step: int) -> None:
            """Record scalar metric entry."""
            recorded_scalars.append((name, data, step))

    class _FakeTF:
        """Typed TensorFlow module double exposing `.summary`."""

        summary = _FakeSummaryNamespace()

    monkeypatch.setattr(log_keras, "tf", _FakeTF())

    log_dir = str(tmp_path / "keras_logs")
    res = log_metrics({"loss": 0.5, "acc": 0.9}, 5, log_dir)
    assert res["status"] == "success"
    assert res["backend"] == "keras"
    assert res.get("action") == "log_metrics"
    assert res["step"] == 5
    assert created_dirs == [log_dir]
    assert len(recorded_scalars) == 2
    assert closed_count == 1


def test_log_metrics_missing_tf(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Keras `log_metrics` raises `DependencyMissingError` when `tf` is `None`."""
    monkeypatch.setattr(log_keras, "tf", None)
    with pytest.raises(Exception, match="TensorFlow dependencies are missing"):
        log_metrics({"loss": 0.5}, 5, "logs")


def test_log_metrics_missing_summary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify Keras `log_metrics` handles `tf` object lacking `summary` attribute."""

    class DummyTF:
        """Stub TensorFlow module without summary attribute."""

    monkeypatch.setattr(log_keras, "tf", DummyTF())
    monkeypatch.setattr("gemma_4_sql.backends.common_logging.get_tensorboard_summary_writer", lambda: None)
    res = log_metrics({"loss": 0.5}, 5, str(tmp_path / "no_summary"), allow_jsonl_fallback=True)
    assert res["status"] == "missing_summary_attr"


def test_log_metrics_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Keras `log_metrics` propagates `RuntimeError` when summary writer creation fails."""

    class _FailingSummary:
        """Summary namespace raising `RuntimeError` on `create_file_writer`."""

        def create_file_writer(self, _logdir: str) -> None:
            """Raise `RuntimeError`.

            Raises:
                RuntimeError: Always raised to simulate summary writer failure.

            """
            raise RuntimeError("TF error")

        def scalar(self, _name: str, _data: float, _step: int) -> None:
            """No-op scalar."""

    class _FailingTF:
        """Typed TensorFlow double with `_FailingSummary`."""

        summary = _FailingSummary()

    _FailingSummary().scalar("unused", 0.0, 0)
    monkeypatch.setattr(log_keras, "tf", _FailingTF())

    with pytest.raises(RuntimeError, match="TF error"):
        log_metrics({"loss": 0.5}, 5, "logs")

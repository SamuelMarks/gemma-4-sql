"""Tests for Logging SDK module."""

import importlib.util
import json
from pathlib import Path

import pytest

from gemma_4_sql.exceptions import ConfigurationError, DependencyMissingError
from gemma_4_sql.sdk.logging import log_metrics

has_tf = importlib.util.find_spec("tensorflow") is not None


def test_log_metrics_jax(tmp_path: Path) -> None:
    """Verify log metrics jax behavior and JSONL fallback file persistence."""
    log_dir = str(tmp_path / "jax_logs")
    res = log_metrics({"loss": 0.5}, 10, log_dir=log_dir, backend="jax")
    assert res["backend"] == "jax"
    assert res.get("action") == "log_metrics"
    assert res.get("step") == 10
    assert res.get("metrics") == {"loss": 0.5}
    fallback_file = res.get("fallback_file")
    assert fallback_file is not None
    lines = Path(fallback_file).read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["step"] == 10
    assert record["metrics"] == {"loss": 0.5}


def test_log_metrics_pytorch(tmp_path: Path) -> None:
    """Verify log metrics pytorch behavior."""
    res = log_metrics({"loss": 0.5}, 10, log_dir=str(tmp_path / "pt_logs"), backend="pytorch")
    assert res["backend"] == "pytorch"
    assert res.get("metrics") == {"loss": 0.5}


def test_log_metrics_keras(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify log metrics keras behavior with a typed TensorFlow summary writer stub and JSONL fallback verification."""
    from contextlib import AbstractContextManager, nullcontext

    import gemma_4_sql.backends.keras.logging as klog

    recorded_scalars: list[tuple[str, float, int]] = []
    closed_writers: list[str] = []

    class _FakeTFWriter:
        """Typed TensorFlow summary file writer stub."""

        def __init__(self, log_dir: str) -> None:
            """Store log directory."""
            self.log_dir = log_dir

        def as_default(self) -> AbstractContextManager[None]:
            """Return nullcontext for `with writer.as_default():`."""
            return nullcontext()

        def close(self) -> None:
            """Record writer closure."""
            closed_writers.append(self.log_dir)

    class _FakeTFSummary:
        """Typed `tf.summary` namespace stub."""

        @staticmethod
        def create_file_writer(log_dir: str) -> _FakeTFWriter:
            """Create a `_FakeTFWriter` instance."""
            return _FakeTFWriter(log_dir)

        @staticmethod
        def scalar(name: str, data: float, step: int) -> None:
            """Record a scalar metric event."""
            recorded_scalars.append((name, float(data), int(step)))

    class _FakeTFModule:
        """Typed `tensorflow` module stub exposing `summary`."""

        summary = _FakeTFSummary

    monkeypatch.setattr(klog, "tf", _FakeTFModule)
    log_dir = str(tmp_path / "keras_logs")
    res = log_metrics({"loss": 0.5}, 10, log_dir=log_dir, backend="keras")
    assert res["backend"] == "keras"
    assert res.get("metrics") == {"loss": 0.5}
    assert recorded_scalars == [("loss", 0.5, 10)]
    assert closed_writers == [log_dir]

    class _FakeTFWithoutSummary:
        """Typed `tensorflow` stub without `summary` attribute to trigger JSONL fallback."""

    monkeypatch.setattr(klog, "tf", _FakeTFWithoutSummary())
    fallback_dir = str(tmp_path / "keras_fallback_logs")
    res_fb = log_metrics({"loss": 0.25}, 20, log_dir=fallback_dir, backend="keras")
    assert res_fb["status"] == "missing_summary_attr"
    fallback_file = res_fb.get("fallback_file")
    assert fallback_file is not None
    payload = json.loads(Path(fallback_file).read_text(encoding="utf-8").strip().splitlines()[0])
    assert payload["step"] == 20
    assert payload["metrics"] == {"loss": 0.25}


def test_log_metrics_maxtext() -> None:
    """Test logging with maxtext."""
    with pytest.raises(DependencyMissingError):
        log_metrics({"loss": 0.5}, 10, backend="maxtext")


def test_log_metrics_invalid() -> None:
    """Verify log_metrics raises ConfigurationError for an unknown backend."""
    with pytest.raises(ConfigurationError, match="Unknown backend: unknown"):
        log_metrics({"loss": 0.5}, 10, backend="unknown")

"""Unit and behavioral tests for common logging."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Never

import pytest

from gemma_4_sql.backends.common_logging import log_metrics_wrapper


class MockWriterWithClose:
    """Test stub for `MockWriterWithClose`."""

    def __init__(self, log_dir: str) -> None:
        """Initialize test stub instance."""
        self.log_dir = log_dir
        self.scalars: list[tuple[str, float, int]] = []
        self.closed = False

    def add_scalar(self, tag: str, scalar_value: float, global_step: int | None = None, **_kwargs: float | str | bool | None) -> None:
        """Provide add scalar test helper."""
        self.scalars.append((tag, scalar_value, int(global_step or 0)))

    def flush(self) -> None:
        """Provide flush test helper."""

    def close(self) -> None:
        """Provide close test helper."""
        self.closed = True


class MockWriterWithoutClose:
    """Test stub for `MockWriterWithoutClose`."""

    def __init__(self, log_dir: str) -> None:
        """Initialize test stub instance."""
        self.log_dir = log_dir
        self.scalars: list[tuple[str, float, int]] = []

    def add_scalar(self, tag: str, scalar_value: float, global_step: int | None = None, **_kwargs: float | str | bool | None) -> None:
        """Provide add scalar test helper."""
        self.scalars.append((tag, scalar_value, int(global_step or 0)))

    def flush(self) -> None:
        """Provide flush test helper."""

    def close(self) -> None:
        """Provide close test helper."""


def test_log_metrics_wrapper_with_writer_with_close(tmp_path: Path) -> None:
    """Verify log metrics wrapper with writer with close behavior."""
    log_dir = str(tmp_path / "logs")
    metrics = {"loss": 0.5, "accuracy": 0.9}

    writer_instances: list[MockWriterWithClose] = []

    def _factory_with_close(log_dir: str = "logs", **_kwargs: str | int | bool | None) -> MockWriterWithClose:
        """Instantiate and record a MockWriterWithClose."""
        inst = MockWriterWithClose(log_dir)
        writer_instances.append(inst)
        return inst

    result = log_metrics_wrapper(
        backend_name="test_backend",
        metrics=metrics,
        step=10,
        log_dir=log_dir,
        summary_writer_cls=_factory_with_close,
        extra_fields={"run_id": "123"},
        step_duration_s=1.5,
    )

    assert result["status"] == "success"
    assert result["backend"] == "test_backend"
    assert result.get("step") == 10
    assert result.get("metrics") == metrics
    assert result.get("step_duration_s") == 1.5
    assert result.get("run_id") == "123"

    assert len(writer_instances) == 1
    writer = writer_instances[0]
    assert writer.closed is True
    assert len(writer.scalars) == 2
    assert ("loss", 0.5, 10) in writer.scalars
    assert ("accuracy", 0.9, 10) in writer.scalars


def test_log_metrics_wrapper_with_writer_without_close(tmp_path: Path) -> None:
    """Verify log metrics wrapper with writer without close behavior."""
    log_dir = str(tmp_path / "logs")
    metrics = {"loss": 0.3}

    writer_instances: list[MockWriterWithoutClose] = []

    def _factory_without_close(log_dir: str = "logs", **_kwargs: str | int | bool | None) -> MockWriterWithoutClose:
        """Instantiate and record a MockWriterWithoutClose."""
        inst = MockWriterWithoutClose(log_dir)
        writer_instances.append(inst)
        return inst

    result = log_metrics_wrapper(
        backend_name="test_backend",
        metrics=metrics,
        step=5,
        log_dir=log_dir,
        summary_writer_cls=_factory_without_close,
    )

    assert result["status"] == "success"
    assert "step_duration_s" not in result
    assert "fallback_file" not in result

    assert len(writer_instances) == 1
    writer = writer_instances[0]
    assert len(writer.scalars) == 1
    assert ("loss", 0.3, 5) in writer.scalars


def test_log_metrics_wrapper_without_writer_fallback(tmp_path: Path) -> None:
    """Verify log metrics wrapper without writer fallback behavior."""
    log_dir = str(tmp_path / "logs")
    metrics = {"val_loss": 0.8}

    result = log_metrics_wrapper(
        backend_name="test_backend2",
        metrics=metrics,
        step=20,
        log_dir=log_dir,
        summary_writer_cls=None,
        extra_fields={"model": "gemma"},
        step_duration_s=2.0,
    )

    assert result["status"] == "backend_unavailable"
    assert result.get("step_duration_s") == 2.0
    assert result.get("model") == "gemma"
    assert "fallback_file" in result

    fallback_file = Path(str(result.get("fallback_file")))
    assert fallback_file.exists()
    assert fallback_file.parent.name == "logs"

    with open(fallback_file) as f:
        lines = f.readlines()

    assert len(lines) == 1
    record = json.loads(lines[0])

    assert record["backend"] == "test_backend2"
    assert record["step"] == 20
    assert record["metrics"] == metrics
    assert record["step_duration_s"] == 2.0
    assert record["model"] == "gemma"
    assert "timestamp" in record


def test_log_metrics_wrapper_without_writer_fallback_minimal(tmp_path: Path) -> None:
    """Verify log metrics wrapper without writer fallback minimal behavior."""
    log_dir = str(tmp_path / "logs_minimal")
    metrics = {"val_loss": 1.2}

    result = log_metrics_wrapper(
        backend_name="test_backend3",
        metrics=metrics,
        step=30,
        log_dir=log_dir,
        summary_writer_cls=None,
    )

    assert result["status"] == "backend_unavailable"
    assert "step_duration_s" not in result
    assert "fallback_file" in result

    fallback_file = Path(str(result.get("fallback_file")))
    assert fallback_file.exists()

    with open(fallback_file) as f:
        lines = f.readlines()

    assert len(lines) == 1
    record = json.loads(lines[0])

    assert record["backend"] == "test_backend3"
    assert record["step"] == 30
    assert record["metrics"] == metrics
    assert "step_duration_s" not in record


def test_log_metrics_wrapper_writer_closes_on_error(tmp_path: Path) -> None:
    """Verify log metrics wrapper writer closes on error behavior."""
    log_dir = str(tmp_path / "logs_err")
    metrics = {"loss": 0.1}

    writer_instances: list[MockWriterWithClose] = []

    class FailingWriterWithClose(MockWriterWithClose):
        """Test stub for `FailingWriterWithClose`."""

        def __init__(self, log_dir: str = "logs", **_kwargs: str | int | bool | None) -> None:
            """Initialize test stub instance."""
            super().__init__(log_dir)
            writer_instances.append(self)

        def add_scalar(self, tag: str, scalar_value: float, global_step: int | None = None, **_kwargs: float | str | bool | None) -> Never:
            """Provide add scalar test helper."""
            _ = (tag, scalar_value, global_step)
            raise ValueError("Intentional error")

    with pytest.raises(ValueError, match="Intentional error"):
        log_metrics_wrapper(
            backend_name="test",
            metrics=metrics,
            step=1,
            log_dir=log_dir,
            summary_writer_cls=FailingWriterWithClose,
        )

    assert len(writer_instances) == 1
    writer = writer_instances[0]
    assert writer.closed is True

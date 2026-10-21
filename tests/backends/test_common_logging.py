"""Unit and behavioral tests for common logging."""

import json
from pathlib import Path

import pytest

from gemma_4_sql.backends.common_logging import log_metrics_wrapper


class MockWriterWithClose:
    """Execute and verify `MockWriterWithClose` behavior and invariants."""

    def __init__(self, log_dir):
        """Execute and verify `__init__` behavior and invariants."""
        self.log_dir = log_dir
        self.scalars = []
        self.closed = False

    def add_scalar(self, tag, scalar_value, global_step):
        """Execute and verify `add_scalar` behavior and invariants."""
        self.scalars.append((tag, scalar_value, global_step))

    def close(self):
        """Execute and verify `close` behavior and invariants."""
        self.closed = True


class MockWriterWithoutClose:
    """Execute and verify `MockWriterWithoutClose` behavior and invariants."""

    def __init__(self, log_dir):
        """Execute and verify `__init__` behavior and invariants."""
        self.log_dir = log_dir
        self.scalars = []

    def add_scalar(self, tag, scalar_value, global_step):
        """Execute and verify `add_scalar` behavior and invariants."""
        self.scalars.append((tag, scalar_value, global_step))


def test_log_metrics_wrapper_with_writer_with_close(tmp_path):
    """Execute and verify `test_log_metrics_wrapper_with_writer_with_close` behavior and invariants."""
    log_dir = str(tmp_path / "logs")
    metrics = {"loss": 0.5, "accuracy": 0.9}

    # We use a factory that returns our mock writer
    writer_instances = []

    class FactoryWithClose:
        """Execute and verify `FactoryWithClose` behavior and invariants."""

        def __new__(cls, log_dir):
            """Execute and verify `__new__` behavior and invariants."""
            inst = MockWriterWithClose(log_dir)
            writer_instances.append(inst)
            return inst

    result = log_metrics_wrapper(backend_name="test_backend", metrics=metrics, step=10, log_dir=log_dir, summary_writer_cls=FactoryWithClose, extra_fields={"run_id": "123"}, step_duration_s=1.5)

    assert result["status"] == "success"
    assert result["backend"] == "test_backend"
    assert result["step"] == 10
    assert result["metrics"] == metrics
    assert result["step_duration_s"] == 1.5
    assert result["run_id"] == "123"

    assert len(writer_instances) == 1
    writer = writer_instances[0]
    assert writer.closed is True
    assert len(writer.scalars) == 2
    assert ("loss", 0.5, 10) in writer.scalars
    assert ("accuracy", 0.9, 10) in writer.scalars


def test_log_metrics_wrapper_with_writer_without_close(tmp_path):
    """Execute and verify `test_log_metrics_wrapper_with_writer_without_close` behavior and invariants."""
    log_dir = str(tmp_path / "logs")
    metrics = {"loss": 0.3}

    writer_instances = []

    class FactoryWithoutClose:
        """Execute and verify `FactoryWithoutClose` behavior and invariants."""

        def __new__(cls, log_dir):
            """Execute and verify `__new__` behavior and invariants."""
            inst = MockWriterWithoutClose(log_dir)
            writer_instances.append(inst)
            return inst

    result = log_metrics_wrapper(
        backend_name="test_backend",
        metrics=metrics,
        step=5,
        log_dir=log_dir,
        summary_writer_cls=FactoryWithoutClose,
    )

    assert result["status"] == "success"
    assert "step_duration_s" not in result
    assert "fallback_file" not in result

    assert len(writer_instances) == 1
    writer = writer_instances[0]
    assert len(writer.scalars) == 1
    assert ("loss", 0.3, 5) in writer.scalars


def test_log_metrics_wrapper_without_writer_fallback(tmp_path):
    """Execute and verify `test_log_metrics_wrapper_without_writer_fallback` behavior and invariants."""
    log_dir = str(tmp_path / "logs")
    metrics = {"val_loss": 0.8}

    result = log_metrics_wrapper(backend_name="test_backend2", metrics=metrics, step=20, log_dir=log_dir, summary_writer_cls=None, extra_fields={"model": "gemma"}, step_duration_s=2.0)

    assert result["status"] == "backend_unavailable"
    assert result["step_duration_s"] == 2.0
    assert result["model"] == "gemma"
    assert "fallback_file" in result

    fallback_file = Path(result["fallback_file"])
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


def test_log_metrics_wrapper_without_writer_fallback_minimal(tmp_path):
    """Execute and verify `test_log_metrics_wrapper_without_writer_fallback_minimal` behavior and invariants."""
    log_dir = str(tmp_path / "logs_minimal")
    metrics = {"val_loss": 1.2}

    result = log_metrics_wrapper(backend_name="test_backend3", metrics=metrics, step=30, log_dir=log_dir, summary_writer_cls=None)

    assert result["status"] == "backend_unavailable"
    assert "step_duration_s" not in result
    assert "fallback_file" in result

    fallback_file = Path(result["fallback_file"])
    assert fallback_file.exists()

    with open(fallback_file) as f:
        lines = f.readlines()

    assert len(lines) == 1
    record = json.loads(lines[0])

    assert record["backend"] == "test_backend3"
    assert record["step"] == 30
    assert record["metrics"] == metrics
    assert "step_duration_s" not in record


def test_log_metrics_wrapper_writer_closes_on_error(tmp_path):
    """Execute and verify `test_log_metrics_wrapper_writer_closes_on_error` behavior and invariants."""
    log_dir = str(tmp_path / "logs_err")
    metrics = {"loss": 0.1}

    writer_instances = []

    class FailingWriterWithClose(MockWriterWithClose):
        """Execute and verify `FailingWriterWithClose` behavior and invariants."""

        def __init__(self, log_dir):
            """Execute and verify `__init__` behavior and invariants."""
            super().__init__(log_dir)
            writer_instances.append(self)

        def add_scalar(self, tag, scalar_value, global_step):
            """Execute and verify `add_scalar` behavior and invariants."""
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

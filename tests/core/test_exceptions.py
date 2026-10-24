"""Unit tests for the unified gemma-4-sql exception hierarchy."""

from __future__ import annotations

import pytest

from gemma_4_sql.exceptions import (
    ConfigurationError,
    DatabaseExecutionError,
    DependencyMissingError,
    ETLError,
    ExecutionStatus,
    ExportError,
    GemmaSQLError,
    InferenceError,
    SchemaExtractionError,
    SQLSafetyViolationError,
    TrainingError,
    UnsupportedQuantizationMethodError,
    is_failed_status,
)


def test_gemma_sql_error_base() -> None:
    """Verify GemmaSQLError stores message, backend, model_name, operation, and cause attributes."""
    cause = ValueError("underlying")
    err = GemmaSQLError(
        "root failure",
        backend="jax",
        model_name="gemma-4-E2B",
        operation="custom_op",
        cause=cause,
    )
    assert str(err) == "root failure"
    assert err.message == "root failure"
    assert err.backend == "jax"
    assert err.model_name == "gemma-4-E2B"
    assert err.operation == "custom_op"
    assert err.cause is cause
    assert isinstance(err, Exception)


def test_dependency_missing_error() -> None:
    """Verify DependencyMissingError inherits from GemmaSQLError and ImportError."""
    err = DependencyMissingError(
        "torch is required",
        dependency_name="torch",
        backend="pytorch",
    )
    assert str(err) == "torch is required"
    assert err.dependency_name == "torch"
    assert err.backend == "pytorch"
    assert isinstance(err, GemmaSQLError)
    assert isinstance(err, ImportError)


def test_unsupported_quantization_method_error() -> None:
    """Verify UnsupportedQuantizationMethodError stores method and supported_methods."""
    err = UnsupportedQuantizationMethodError(
        "fp4 not supported",
        method="fp4",
        supported_methods=("int8", "int4", "awq"),
        backend="mlx",
    )
    assert str(err) == "fp4 not supported"
    assert err.method == "fp4"
    assert err.supported_methods == ("int8", "int4", "awq")
    assert err.backend == "mlx"
    assert isinstance(err, GemmaSQLError)
    assert isinstance(err, ValueError)


@pytest.mark.parametrize(
    ("exc_cls", "std_parent"),
    [
        (InferenceError, RuntimeError),
        (ExportError, RuntimeError),
        (TrainingError, RuntimeError),
        (ETLError, RuntimeError),
        (DatabaseExecutionError, RuntimeError),
        (SQLSafetyViolationError, PermissionError),
        (SchemaExtractionError, ValueError),
        (ConfigurationError, ValueError),
    ],
)
def test_domain_exception_inheritance(
    exc_cls: type[GemmaSQLError],
    std_parent: type[Exception],
) -> None:
    """Verify domain exceptions inherit from both GemmaSQLError and standard exception types."""
    err = exc_cls("domain error", backend="keras")
    assert str(err) == "domain error"
    assert err.backend == "keras"
    assert isinstance(err, GemmaSQLError)
    assert isinstance(err, std_parent)
    if exc_cls is SQLSafetyViolationError:
        assert isinstance(err, DatabaseExecutionError)


def test_structured_operation_exceptions() -> None:
    """Verify TrainingError, InferenceError, and ExportError store default and custom attributes."""
    t_err = TrainingError("sft boom", backend="jax", model_name="gemma-4-E2B")
    assert t_err.backend == "jax"
    assert t_err.model_name == "gemma-4-E2B"
    assert t_err.operation == "train"

    i_err = InferenceError("beam boom", backend="pytorch", model_name="gemma-4-E4B", operation="generate_sql")
    assert i_err.backend == "pytorch"
    assert i_err.model_name == "gemma-4-E4B"
    assert i_err.operation == "generate_sql"

    e_err = ExportError("orbax boom", backend="maxtext", model_name="gemma-4-31B")
    assert e_err.backend == "maxtext"
    assert e_err.model_name == "gemma-4-31B"
    assert e_err.operation == "export"


def test_is_failed_status() -> None:
    """Verify is_failed_status detects ExecutionStatus, failed status strings, and error dictionaries."""
    assert is_failed_status(ExecutionStatus.FAILED) is True
    assert is_failed_status(ExecutionStatus.UNSUPPORTED) is True
    assert is_failed_status(ExecutionStatus.SUCCESS) is False
    assert is_failed_status("failed: boom") is True
    assert is_failed_status("unsupported_method_fp4") is True
    assert is_failed_status("missing_tensorboard") is True
    assert is_failed_status("completed") is False
    assert is_failed_status({"status": "failed: err"}) is True
    assert is_failed_status({"status": ExecutionStatus.ERROR}) is True
    assert is_failed_status({"status": "completed", "error": "boom"}) is True
    assert is_failed_status({"status": "completed"}) is False
    assert is_failed_status({"status": 123}) is False
    assert is_failed_status(123) is False

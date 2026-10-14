"""Unit tests for the unified gemma-4-sql exception hierarchy."""

from __future__ import annotations

import pytest

from gemma_4_sql.exceptions import (
    ConfigurationError,
    DatabaseExecutionError,
    DependencyMissingError,
    ETLError,
    ExportError,
    GemmaSQLError,
    InferenceError,
    SchemaExtractionError,
    SQLSafetyViolationError,
    TrainingError,
    UnsupportedQuantizationMethodError,
)


def test_gemma_sql_error_base() -> None:
    """Verify GemmaSQLError stores message, backend, and cause attributes."""
    cause = ValueError("underlying")
    err = GemmaSQLError("root failure", backend="jax", cause=cause)
    assert str(err) == "root failure"
    assert err.message == "root failure"
    assert err.backend == "jax"
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

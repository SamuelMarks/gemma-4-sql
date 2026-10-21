"""Unified custom exception hierarchy for gemma-4-sql."""

from __future__ import annotations


class GemmaSQLError(Exception):
    """Root base exception for all gemma-4-sql errors.

    Attributes:
        message: Human-readable error message.
        backend: Optional backend identifier where the error originated.
        cause: Optional underlying exception that triggered this error.

    """

    def __init__(
        self,
        message: str = "",
        *,
        backend: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize GemmaSQLError with structured context.

        Args:
            message: Human-readable error description.
            backend: Optional name of the backend where the error occurred.
            cause: Optional underlying exception cause.

        """
        super().__init__(message)
        self.message = message
        self.backend = backend
        self.cause = cause


class DependencyMissingError(GemmaSQLError, RuntimeError, ImportError):
    """Raised when a required optional backend dependency is missing.

    Attributes:
        dependency_name: Name of the missing package or module.
        backend: Backend requiring the missing package.

    """

    def __init__(
        self,
        message: str = "Required dependency is missing.",
        *,
        dependency_name: str = "",
        backend: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize DependencyMissingError.

        Args:
            message: Human-readable error description.
            dependency_name: Optional package identifier that could not be imported.
            backend: Optional backend identifier requiring the package.
            cause: Optional underlying ImportError cause.

        """
        super().__init__(message, backend=backend, cause=cause)
        self.dependency_name = dependency_name


class UnsupportedQuantizationMethodError(GemmaSQLError, ValueError):
    """Raised when a quantization method is unsupported by the target backend.

    Attributes:
        method: The requested quantization method string.
        supported_methods: Tuple of valid quantization method strings.

    """

    def __init__(
        self,
        message: str = "Unsupported quantization method.",
        *,
        method: str = "",
        supported_methods: tuple[str, ...] = (),
        backend: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize UnsupportedQuantizationMethodError.

        Args:
            message: Human-readable error description.
            method: The unsupported quantization method name.
            supported_methods: Tuple of supported quantization method names.
            backend: Optional backend identifier.
            cause: Optional underlying exception cause.

        """
        super().__init__(message, backend=backend, cause=cause)
        self.method = method
        self.supported_methods = supported_methods


class InferenceError(GemmaSQLError, RuntimeError):
    """Raised when model inference or SQL generation fails."""


class ExportError(GemmaSQLError, RuntimeError, ValueError):
    """Raised when checkpoint serialization or model export fails."""


class TrainingError(GemmaSQLError, RuntimeError):
    """Raised when pretraining, SFT, post-training, or DPO execution fails."""


class ETLError(GemmaSQLError, RuntimeError):
    """Raised when dataset loading, DuckDB ingestion, or DataLoader construction fails."""


class DatabaseExecutionError(GemmaSQLError, RuntimeError):
    """Raised when database adapter connection or SQL query execution fails."""


class SQLSafetyViolationError(DatabaseExecutionError, PermissionError):
    """Raised when a query violates read-only, DDL, or multi-statement safety guards."""


class SchemaExtractionError(GemmaSQLError, ValueError):
    """Raised when DDL schema extraction or parsing fails."""


class ConfigurationError(GemmaSQLError, ValueError):
    """Raised when hyperparameters, backend names, or configuration parameters are invalid."""


def is_failed_status(result: object) -> bool:
    """Return True if a backend result dictionary or status string indicates execution failure.

    Args:
        result: A backend result mapping (containing `'status'` or `'error'`) or a status string.

    Returns:
        True if the status starts with `'failed'` or a non-empty `'error'` field is present.

    """
    if isinstance(result, str):
        return result.lower().startswith("failed")
    if isinstance(result, dict):
        err = result.get("error")
        if isinstance(err, str) and err:
            return True
        status = result.get("status", "")
        return isinstance(status, str) and status.lower().startswith("failed")
    return False


__all__ = [
    "ConfigurationError",
    "DatabaseExecutionError",
    "DependencyMissingError",
    "ETLError",
    "ExportError",
    "GemmaSQLError",
    "InferenceError",
    "SQLSafetyViolationError",
    "SchemaExtractionError",
    "TrainingError",
    "UnsupportedQuantizationMethodError",
    "is_failed_status",
]

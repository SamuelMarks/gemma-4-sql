"""Unified custom exception hierarchy for gemma-4-sql."""

from __future__ import annotations

from enum import StrEnum


class ExecutionStatus(StrEnum):
    """Canonical execution status values for backend operations."""

    SUCCESS = "success"
    COMPLETED = "completed"
    LOADED = "loaded"
    FAILED = "failed"
    ERROR = "error"
    UNSUPPORTED = "unsupported"
    MISSING_DEPENDENCY = "missing_dependency"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    INJECTED = "injected"
    NO_MATCHING_MODULES = "no_matching_modules"
    EMPTY_DATASET = "empty_dataset"


class GemmaSQLError(Exception):
    """Root base exception for all gemma-4-sql errors.

    Attributes:
        message: Human-readable error message.
        backend: Optional backend identifier where the error originated.
        model_name: Optional model identifier associated with the operation.
        operation: Optional operation name where the error occurred.
        cause: Optional underlying exception that triggered this error.

    """

    def __init__(
        self,
        message: str = "",
        *,
        backend: str | None = None,
        model_name: str | None = None,
        operation: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize GemmaSQLError with structured context.

        Args:
            message: Human-readable error description.
            backend: Optional name of the backend where the error occurred.
            model_name: Optional name of the model associated with the operation.
            operation: Optional name of the operation that failed.
            cause: Optional underlying exception cause.

        """
        super().__init__(message)
        self.message = message
        self.backend = backend
        self.model_name = model_name
        self.operation = operation
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


class InferenceError(GemmaSQLError, RuntimeError, ValueError):
    """Raised when model inference or SQL generation fails.

    Attributes:
        backend: Optional backend identifier where inference failed.
        model_name: Optional model identifier used during inference.
        operation: Operation name (defaults to `'inference'`).

    """

    def __init__(
        self,
        message: str = "Inference failed.",
        *,
        backend: str | None = None,
        model_name: str | None = None,
        operation: str | None = "inference",
        cause: Exception | None = None,
    ) -> None:
        """Initialize InferenceError with structured backend and model context.

        Args:
            message: Human-readable error description.
            backend: Optional backend identifier where inference failed.
            model_name: Optional model identifier used during inference.
            operation: Optional operation identifier (defaults to `'inference'`).
            cause: Optional underlying exception cause.

        """
        super().__init__(
            message,
            backend=backend,
            model_name=model_name,
            operation=operation,
            cause=cause,
        )


class ExportError(GemmaSQLError, RuntimeError, ValueError):
    """Raised when checkpoint serialization or model export fails.

    Attributes:
        backend: Optional backend identifier where export failed.
        model_name: Optional model identifier being exported.
        operation: Operation name (defaults to `'export'`).

    """

    def __init__(
        self,
        message: str = "Model export failed.",
        *,
        backend: str | None = None,
        model_name: str | None = None,
        operation: str | None = "export",
        cause: Exception | None = None,
    ) -> None:
        """Initialize ExportError with structured backend and model context.

        Args:
            message: Human-readable error description.
            backend: Optional backend identifier where export failed.
            model_name: Optional model identifier being exported.
            operation: Optional operation identifier (defaults to `'export'`).
            cause: Optional underlying exception cause.

        """
        super().__init__(
            message,
            backend=backend,
            model_name=model_name,
            operation=operation,
            cause=cause,
        )


class TrainingError(GemmaSQLError, RuntimeError, ValueError):
    """Raised when pretraining, SFT, post-training, PEFT, or DPO execution fails.

    Attributes:
        backend: Optional backend identifier where training failed.
        model_name: Optional model identifier being trained.
        operation: Operation name (defaults to `'train'`).

    """

    def __init__(
        self,
        message: str = "Training failed.",
        *,
        backend: str | None = None,
        model_name: str | None = None,
        operation: str | None = "train",
        cause: Exception | None = None,
    ) -> None:
        """Initialize TrainingError with structured backend and model context.

        Args:
            message: Human-readable error description.
            backend: Optional backend identifier where training failed.
            model_name: Optional model identifier being trained.
            operation: Optional operation identifier (defaults to `'train'`).
            cause: Optional underlying exception cause.

        """
        super().__init__(
            message,
            backend=backend,
            model_name=model_name,
            operation=operation,
            cause=cause,
        )


class ChatError(InferenceError):
    """Raised when a multi-turn conversational SQL chat turn fails."""


class EvaluationError(GemmaSQLError, ValueError):
    """Raised when model evaluation or metric computation fails."""


class ServingError(GemmaSQLError, RuntimeError):
    """Raised when continuous batching or model serving initialization fails."""


class BenchmarkError(GemmaSQLError, RuntimeError):
    """Raised when hardware benchmarking execution fails."""


class PEFTError(TrainingError):
    """Raised when Parameter-Efficient Fine-Tuning (PEFT / LoRA) injection fails."""


class QuantizationError(GemmaSQLError, RuntimeError, ValueError):
    """Raised when model quantization execution fails."""


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


_FAILED_STATUS_VALUES: frozenset[str] = frozenset({
    ExecutionStatus.FAILED.value,
    ExecutionStatus.ERROR.value,
    ExecutionStatus.UNSUPPORTED.value,
    ExecutionStatus.MISSING_DEPENDENCY.value,
    ExecutionStatus.BACKEND_UNAVAILABLE.value,
})


def resolve_raise_on_error(config: object | None = None, *, raise_on_error: bool | None = None, kwargs: dict[str, object] | None = None) -> bool:
    """Resolve whether an operation should raise a `GemmaSQLError` on failure.

    Args:
        config: Optional configuration object (`TrainingConfig`, `DPOConfig`, `ETLConfig`).
        raise_on_error: Explicit boolean override if provided by the caller.
        kwargs: Optional keyword arguments dictionary that may contain `'raise_on_error'`.

    Returns:
        `True` if errors should be raised rather than returned in the status dictionary.

    """
    if raise_on_error is not None:
        return bool(raise_on_error)
    if kwargs is not None and "raise_on_error" in kwargs:
        return bool(kwargs["raise_on_error"])
    if config is not None:
        if bool(getattr(config, "raise_on_error", False)):
            return True
        extra = getattr(config, "extra_kwargs", None)
        if isinstance(extra, dict) and "raise_on_error" in extra:
            return bool(extra["raise_on_error"])
    return False


def is_failed_status(result: object) -> bool:
    """Return True if a backend result dictionary, status enum, or status string indicates failure.

    Args:
        result: A backend result mapping (containing `'status'` or `'error'`), `ExecutionStatus`, or status string.

    Returns:
        True if the status represents a failed, unsupported, or missing-dependency outcome.

    """
    if isinstance(result, ExecutionStatus):
        return result.value in _FAILED_STATUS_VALUES
    if isinstance(result, str):
        norm = result.strip().lower()
        return norm in _FAILED_STATUS_VALUES or norm.startswith(("failed", "error:", "unsupported_", "missing_", "backend_unavailable"))
    if isinstance(result, dict):
        err = result.get("error")
        if isinstance(err, str) and err:
            return True
        status = result.get("status", "")
        if isinstance(status, (str, ExecutionStatus)):
            return is_failed_status(status)
    return False


__all__ = [
    "BenchmarkError",
    "ChatError",
    "ConfigurationError",
    "DatabaseExecutionError",
    "DependencyMissingError",
    "ETLError",
    "EvaluationError",
    "ExecutionStatus",
    "ExportError",
    "GemmaSQLError",
    "InferenceError",
    "PEFTError",
    "QuantizationError",
    "SQLSafetyViolationError",
    "SchemaExtractionError",
    "ServingError",
    "TrainingError",
    "UnsupportedQuantizationMethodError",
    "is_failed_status",
    "resolve_raise_on_error",
]

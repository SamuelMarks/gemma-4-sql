"""Unified custom exception hierarchy for gemma-4-sql."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from enum import StrEnum
from typing import Protocol, Union, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class StringKeyedMappingProtocol(Protocol):
    """Protocol satisfied by all string-keyed mappings and TypedDicts."""

    def __iter__(self) -> Iterator[str]:
        """Iterate over the string keys of the mapping."""
        ...

    def __len__(self) -> int:
        """Return the number of entries in the mapping."""
        ...


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


ResultStatus = ExecutionStatus


@runtime_checkable
class ErrorPolicyConfigProtocol(Protocol):
    """Protocol for configuration objects carrying `raise_on_error`."""

    @property
    def raise_on_error(self) -> bool:
        """Provide whether operational errors should be raised immediately."""
        ...


@runtime_checkable
class ExtraKwargsCarrierProtocol(Protocol):
    """Protocol for configuration objects carrying `extra_kwargs`."""

    @property
    def extra_kwargs(self) -> Mapping[str, str | int | float | bool | None] | None:
        """Provide extra keyword arguments mapping."""
        ...


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
        model_name: str | None = None,
        operation: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize DependencyMissingError.

        Args:
            message: Human-readable error description.
            dependency_name: Optional package identifier that could not be imported.
            backend: Optional backend identifier requiring the package.
            model_name: Optional model identifier associated with the operation.
            operation: Optional operation identifier.
            cause: Optional underlying ImportError cause.

        """
        super().__init__(
            message,
            backend=backend,
            model_name=model_name,
            operation=operation,
            cause=cause,
        )
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
        model_name: str | None = None,
        operation: str | None = "quantize",
        cause: Exception | None = None,
    ) -> None:
        """Initialize UnsupportedQuantizationMethodError.

        Args:
            message: Human-readable error description.
            method: The unsupported quantization method name.
            supported_methods: Tuple of supported quantization method names.
            backend: Optional backend identifier.
            model_name: Optional model identifier.
            operation: Optional operation identifier (defaults to `'quantize'`).
            cause: Optional underlying exception cause.

        """
        super().__init__(
            message,
            backend=backend,
            model_name=model_name,
            operation=operation,
            cause=cause,
        )
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

    def __init__(
        self,
        message: str = "Chat turn failed.",
        *,
        backend: str | None = None,
        model_name: str | None = None,
        operation: str | None = "chat",
        cause: Exception | None = None,
    ) -> None:
        """Initialize ChatError with structured backend and model context.

        Args:
            message: Human-readable error description.
            backend: Optional backend identifier where chat failed.
            model_name: Optional model identifier used during chat.
            operation: Optional operation identifier (defaults to `'chat'`).
            cause: Optional underlying exception cause.

        """
        super().__init__(
            message,
            backend=backend,
            model_name=model_name,
            operation=operation,
            cause=cause,
        )


class EvaluationError(GemmaSQLError, ValueError):
    """Raised when model evaluation or metric computation fails."""

    def __init__(
        self,
        message: str = "Evaluation failed.",
        *,
        backend: str | None = None,
        model_name: str | None = None,
        operation: str | None = "evaluate",
        cause: Exception | None = None,
    ) -> None:
        """Initialize EvaluationError with structured backend and model context.

        Args:
            message: Human-readable error description.
            backend: Optional backend identifier where evaluation failed.
            model_name: Optional model identifier being evaluated.
            operation: Optional operation identifier (defaults to `'evaluate'`).
            cause: Optional underlying exception cause.

        """
        super().__init__(
            message,
            backend=backend,
            model_name=model_name,
            operation=operation,
            cause=cause,
        )


class ServingError(GemmaSQLError, RuntimeError):
    """Raised when continuous batching or model serving initialization fails."""

    def __init__(
        self,
        message: str = "Serving failed.",
        *,
        backend: str | None = None,
        model_name: str | None = None,
        operation: str | None = "serve",
        cause: Exception | None = None,
    ) -> None:
        """Initialize ServingError with structured backend and model context.

        Args:
            message: Human-readable error description.
            backend: Optional backend identifier where serving failed.
            model_name: Optional model identifier being served.
            operation: Optional operation identifier (defaults to `'serve'`).
            cause: Optional underlying exception cause.

        """
        super().__init__(
            message,
            backend=backend,
            model_name=model_name,
            operation=operation,
            cause=cause,
        )


class BenchmarkError(GemmaSQLError, RuntimeError):
    """Raised when hardware benchmarking execution fails."""

    def __init__(
        self,
        message: str = "Benchmark failed.",
        *,
        backend: str | None = None,
        model_name: str | None = None,
        operation: str | None = "benchmark",
        cause: Exception | None = None,
    ) -> None:
        """Initialize BenchmarkError with structured backend and model context.

        Args:
            message: Human-readable error description.
            backend: Optional backend identifier where benchmarking failed.
            model_name: Optional model identifier being benchmarked.
            operation: Optional operation identifier (defaults to `'benchmark'`).
            cause: Optional underlying exception cause.

        """
        super().__init__(
            message,
            backend=backend,
            model_name=model_name,
            operation=operation,
            cause=cause,
        )


class PEFTError(TrainingError):
    """Raised when Parameter-Efficient Fine-Tuning (PEFT / LoRA) injection fails."""

    def __init__(
        self,
        message: str = "PEFT injection failed.",
        *,
        backend: str | None = None,
        model_name: str | None = None,
        operation: str | None = "peft",
        cause: Exception | None = None,
    ) -> None:
        """Initialize PEFTError with structured backend and model context.

        Args:
            message: Human-readable error description.
            backend: Optional backend identifier where PEFT failed.
            model_name: Optional model identifier being adapted.
            operation: Optional operation identifier (defaults to `'peft'`).
            cause: Optional underlying exception cause.

        """
        super().__init__(
            message,
            backend=backend,
            model_name=model_name,
            operation=operation,
            cause=cause,
        )


class QuantizationError(GemmaSQLError, RuntimeError, ValueError):
    """Raised when model quantization execution fails."""

    def __init__(
        self,
        message: str = "Quantization failed.",
        *,
        backend: str | None = None,
        model_name: str | None = None,
        operation: str | None = "quantize",
        cause: Exception | None = None,
    ) -> None:
        """Initialize QuantizationError with structured backend and model context.

        Args:
            message: Human-readable error description.
            backend: Optional backend identifier where quantization failed.
            model_name: Optional model identifier being quantized.
            operation: Optional operation identifier (defaults to `'quantize'`).
            cause: Optional underlying exception cause.

        """
        super().__init__(
            message,
            backend=backend,
            model_name=model_name,
            operation=operation,
            cause=cause,
        )


class ETLError(GemmaSQLError, RuntimeError):
    """Raised when dataset loading, DuckDB ingestion, or DataLoader construction fails."""

    def __init__(
        self,
        message: str = "ETL failed.",
        *,
        backend: str | None = None,
        model_name: str | None = None,
        operation: str | None = "etl",
        cause: Exception | None = None,
    ) -> None:
        """Initialize ETLError with structured backend and model context.

        Args:
            message: Human-readable error description.
            backend: Optional backend identifier where ETL failed.
            model_name: Optional model or dataset identifier.
            operation: Optional operation identifier (defaults to `'etl'`).
            cause: Optional underlying exception cause.

        """
        super().__init__(
            message,
            backend=backend,
            model_name=model_name,
            operation=operation,
            cause=cause,
        )


class DatabaseExecutionError(GemmaSQLError, RuntimeError):
    """Raised when database adapter connection or SQL query execution fails."""


class SQLSafetyViolationError(DatabaseExecutionError, PermissionError):
    """Raised when a query violates read-only, DDL, or multi-statement safety guards."""


class SchemaExtractionError(GemmaSQLError, ValueError):
    """Raised when DDL schema extraction or parsing fails."""


class ConfigurationError(GemmaSQLError, ValueError):
    """Raised when hyperparameters, backend names, or configuration parameters are invalid."""


class MultimodalProcessingError(GemmaSQLError, ValueError):
    """Raised when image or audio input decoding, validation, or feature extraction fails."""


OPERATIONAL_EXCEPTIONS: tuple[type[Exception], ...] = (
    GemmaSQLError,
    RuntimeError,
    ValueError,
    TypeError,
    OSError,
    LookupError,
    ArithmeticError,
    AttributeError,
    ImportError,
)


_FAILED_STATUS_VALUES: frozenset[str] = frozenset({
    ExecutionStatus.FAILED.value,
    ExecutionStatus.ERROR.value,
    ExecutionStatus.UNSUPPORTED.value,
    ExecutionStatus.MISSING_DEPENDENCY.value,
    ExecutionStatus.BACKEND_UNAVAILABLE.value,
})

_FAILED_STATUS_PREFIXES: tuple[str, ...] = (
    "failed:",
    "error:",
    "unsupported_",
    "missing_",
)

StatusScalar = Union[ExecutionStatus, str, int, float, bool, None]
type StatusValue = Union[
    StatusScalar,
    Sequence["StatusValue"],
    Mapping[str, "StatusValue"],
]
StatusMapping = Mapping[str, StatusValue]
StatusInput = Union[
    ExecutionStatus,
    str,
    int,
    float,
    bool,
    StatusMapping,
    StringKeyedMappingProtocol,
    Sequence[StatusValue],
    None,
]


def resolve_raise_on_error(
    config: ErrorPolicyConfigProtocol | ExtraKwargsCarrierProtocol | StatusInput = None,
    *,
    raise_on_error: bool | None = None,
    kwargs: StatusMapping | None = None,
) -> bool:
    """Resolve whether an operation should raise a `GemmaSQLError` subclass on runtime failure.

    Error Contract:
        - `DependencyMissingError`, `ConfigurationError`, and `UnsupportedQuantizationMethodError`
          always raise immediately regardless of `raise_on_error`.
        - Operational runtime errors (`TrainingError`, `InferenceError`, `PEFTError`,
          `QuantizationError`, `ServingError`, `BenchmarkError`, `ExportError`) raise when
          `resolve_raise_on_error(...)` evaluates to `True`, and otherwise populate structured
          `status_code: ExecutionStatus.FAILED`, `error_message: str`, and `status: f"failed: {e}"`
          fields in the returned result dictionary.

    Args:
        config: Optional configuration object (`TrainingConfig`, `DPOConfig`, `ETLConfig`).
        raise_on_error: Explicit boolean override if provided by the caller.
        kwargs: Optional keyword arguments mapping that may contain `'raise_on_error'`.

    Returns:
        `True` if operational errors should be raised rather than returned in the status dictionary.

    """
    if raise_on_error is not None:
        return bool(raise_on_error)
    if kwargs is not None and "raise_on_error" in kwargs:
        return bool(kwargs["raise_on_error"])
    if config is not None:
        if isinstance(config, ErrorPolicyConfigProtocol) and config.raise_on_error:
            return True
        if isinstance(config, ExtraKwargsCarrierProtocol):
            extra = config.extra_kwargs
            if isinstance(extra, Mapping) and "raise_on_error" in extra:
                return bool(extra["raise_on_error"])
    return False


def is_failed_status(result: StatusInput) -> bool:
    """Return True if a backend result dictionary, status enum, or status string indicates failure.

    Checks structured `ExecutionStatus` enum values (`status_code`, `status`) and `error` /
    `error_message` keys first, falling back to canonical prefix matching for status strings.

    Args:
        result: A backend result mapping (containing `'status_code'`, `'error_message'`, `'error'`,
            or `'status'`), `ExecutionStatus`, or status string.

    Returns:
        True if the status represents a failed, unsupported, or missing-dependency outcome.

    """
    if isinstance(result, ExecutionStatus):
        return result.value in _FAILED_STATUS_VALUES
    if isinstance(result, str):
        norm = result.strip().lower()
        return norm in _FAILED_STATUS_VALUES or norm.startswith(_FAILED_STATUS_PREFIXES)
    if isinstance(result, Mapping):
        status_code = result.get("status_code")
        if isinstance(status_code, (str, ExecutionStatus)) and is_failed_status(status_code):
            return True
        err_msg = result.get("error_message")
        if isinstance(err_msg, str) and err_msg:
            return True
        err = result.get("error")
        if isinstance(err, str) and err:
            return True
        status = result.get("status", "")
        if isinstance(status, (str, ExecutionStatus)):
            if status_code is None and is_failed_status(status):
                logger.debug("Legacy failed status dictionary missing 'status_code': %s", status)
                return True
            return is_failed_status(status)
    return False


__all__ = [
    "OPERATIONAL_EXCEPTIONS",
    "BenchmarkError",
    "ChatError",
    "ConfigurationError",
    "DatabaseExecutionError",
    "DependencyMissingError",
    "ETLError",
    "ErrorPolicyConfigProtocol",
    "EvaluationError",
    "ExecutionStatus",
    "ExportError",
    "ExtraKwargsCarrierProtocol",
    "GemmaSQLError",
    "InferenceError",
    "MultimodalProcessingError",
    "PEFTError",
    "QuantizationError",
    "ResultStatus",
    "SQLSafetyViolationError",
    "SchemaExtractionError",
    "ServingError",
    "StatusInput",
    "StatusMapping",
    "StatusScalar",
    "TrainingError",
    "UnsupportedQuantizationMethodError",
    "is_failed_status",
    "resolve_raise_on_error",
]

"""Common model export wrapper and standardized result construction across backends."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from gemma_4_sql.exceptions import (
    OPERATIONAL_EXCEPTIONS,
    ConfigurationError,
    DependencyMissingError,
    ExecutionStatus,
    ExportError,
    is_failed_status,
)
from gemma_4_sql.type_hints import ExportResultDict


def build_export_result(
    backend_name: str,
    model_name: str,
    export_path: str,
    file_path: str | Path,
    status: str,
    export_format: str,
    *,
    status_code: str | None = None,
    error_message: str | None = None,
) -> ExportResultDict:
    """Construct a standardized `ExportResultDict`.

    Args:
        backend_name: Backend identifier ('jax', 'keras', 'maxtext', 'mlx', 'pytorch').
        model_name: Target model name.
        export_path: Destination export directory path.
        file_path: Serialized checkpoint or artifact path.
        status: Descriptive export status string.
        export_format: Serialized format identifier (e.g., `'safetensors'`, `'orbax/saved_model'`).
        status_code: Optional canonical `ExecutionStatus` value.
        error_message: Optional error message if export failed.

    Returns:
        Populated `ExportResultDict`.

    """
    resolved_code = status_code or (ExecutionStatus.FAILED.value if is_failed_status(status) else ExecutionStatus.COMPLETED.value)
    result: ExportResultDict = {
        "backend": backend_name,
        "model": model_name,
        "export_path": export_path,
        "file_path": str(file_path),
        "status": status,
        "status_code": resolved_code,
        "format": export_format,
    }
    if error_message is not None:
        result["error_message"] = error_message
    return result


def run_export_wrapper(
    backend_name: str,
    model_name: str,
    export_path: str,
    export_format: str,
    execute_fn: Callable[[], tuple[str | Path, str]],
    *,
    raise_on_error: bool = True,
) -> ExportResultDict:
    """Execute a backend model export pipeline with standardized directory setup, error wrapping, and result formatting.

    Args:
        backend_name: Backend identifier ('jax', 'keras', 'maxtext', 'mlx', 'pytorch').
        model_name: Target model name.
        export_path: Destination export directory path.
        export_format: Serialized format identifier.
        execute_fn: Zero-argument callable executing the export and returning `(file_path, status)`.
        raise_on_error: Whether to raise `ExportError` on failure (defaults to `True`).

    Returns:
        Structured `ExportResultDict`.

    Raises:
        DependencyMissingError: Re-raised if raised by `execute_fn`.
        ConfigurationError: Re-raised if raised by `execute_fn`.
        ExportError: If `execute_fn` raises an operational exception and `raise_on_error` is True.

    """
    Path(export_path).mkdir(parents=True, exist_ok=True)
    error_message: str | None = None
    try:
        file_path, status = execute_fn()
    except (DependencyMissingError, ConfigurationError):
        raise
    except OPERATIONAL_EXCEPTIONS as e:
        if raise_on_error:
            if isinstance(e, ExportError):
                raise
            raise ExportError(
                f"Failed to export model '{model_name}' on {backend_name}: {e!s}",
                backend=backend_name,
                model_name=model_name,
                operation="export",
                cause=e,
            ) from e
        error_message = str(e)
        file_path = export_path
        status = f"failed: {error_message}"

    return build_export_result(
        backend_name=backend_name,
        model_name=model_name,
        export_path=export_path,
        file_path=file_path,
        status=status,
        export_format=export_format,
        error_message=error_message,
    )


__all__ = ["build_export_result", "run_export_wrapper"]

"""Custom exceptions for gemma-4-sql."""


class DependencyMissingError(Exception):
    """Raised when a required dependency is missing."""


class UnsupportedQuantizationMethodError(ValueError):
    """Raised when a quantization method is unsupported by the backend."""


class InferenceError(RuntimeError):
    """Raised when model inference or generation fails."""


class ExportError(RuntimeError):
    """Raised when checkpoint serialization or model export fails."""

"""Unit tests for the unified gemma-4-sql exception hierarchy."""

from __future__ import annotations

import pytest

from gemma_4_sql.exceptions import (
    BenchmarkError,
    ChatError,
    ConfigurationError,
    DatabaseExecutionError,
    DependencyMissingError,
    ETLError,
    EvaluationError,
    ExecutionStatus,
    ExportError,
    GemmaSQLError,
    InferenceError,
    MultimodalProcessingError,
    PEFTError,
    QuantizationError,
    ResultStatus,
    SchemaExtractionError,
    ServingError,
    SQLSafetyViolationError,
    TrainingError,
    UnsupportedQuantizationMethodError,
    is_failed_status,
    resolve_raise_on_error,
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
    assert ResultStatus is ExecutionStatus


def test_dependency_missing_error() -> None:
    """Verify DependencyMissingError inherits from GemmaSQLError and ImportError."""
    err = DependencyMissingError(
        "torch is required",
        dependency_name="torch",
        backend="pytorch",
        model_name="gemma-4-E2B",
        operation="train",
    )
    assert str(err) == "torch is required"
    assert err.dependency_name == "torch"
    assert err.backend == "pytorch"
    assert err.model_name == "gemma-4-E2B"
    assert err.operation == "train"
    assert isinstance(err, GemmaSQLError)
    assert isinstance(err, ImportError)


def test_unsupported_quantization_method_error() -> None:
    """Verify UnsupportedQuantizationMethodError stores method and supported_methods."""
    err = UnsupportedQuantizationMethodError(
        "fp4 not supported",
        method="fp4",
        supported_methods=("int8", "int4", "awq"),
        backend="mlx",
        model_name="gemma-4-E2B",
    )
    assert str(err) == "fp4 not supported"
    assert err.method == "fp4"
    assert err.supported_methods == ("int8", "int4", "awq")
    assert err.backend == "mlx"
    assert err.model_name == "gemma-4-E2B"
    assert err.operation == "quantize"
    assert isinstance(err, GemmaSQLError)
    assert isinstance(err, ValueError)


@pytest.mark.parametrize(
    ("exc_cls", "std_parent"),
    [
        (InferenceError, RuntimeError),
        (ChatError, InferenceError),
        (ExportError, RuntimeError),
        (TrainingError, RuntimeError),
        (PEFTError, TrainingError),
        (ETLError, RuntimeError),
        (EvaluationError, ValueError),
        (ServingError, RuntimeError),
        (BenchmarkError, RuntimeError),
        (QuantizationError, RuntimeError),
        (DatabaseExecutionError, RuntimeError),
        (SQLSafetyViolationError, PermissionError),
        (SchemaExtractionError, ValueError),
        (ConfigurationError, ValueError),
        (MultimodalProcessingError, ValueError),
    ],
)
def test_domain_exception_inheritance(
    exc_cls: type[GemmaSQLError],
    std_parent: type[Exception],
) -> None:
    """Verify domain exceptions inherit from both GemmaSQLError and standard exception types."""
    err = exc_cls("domain error", backend="keras", model_name="gemma-4-E2B")
    assert str(err) == "domain error"
    assert err.backend == "keras"
    assert err.model_name == "gemma-4-E2B"
    assert isinstance(err, GemmaSQLError)
    assert isinstance(err, std_parent)
    if exc_cls is SQLSafetyViolationError:
        assert isinstance(err, DatabaseExecutionError)


def test_structured_operation_exceptions() -> None:
    """Verify operational exceptions store default and custom backend, model_name, and operation attributes."""
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

    q_err = QuantizationError("quant boom", backend="mlx", model_name="gemma-4-E2B")
    assert q_err.operation == "quantize"

    s_err = ServingError("serve boom", backend="pytorch", model_name="gemma-4-E2B")
    assert s_err.operation == "serve"

    b_err = BenchmarkError("bench boom", backend="jax", model_name="gemma-4-E2B")
    assert b_err.operation == "benchmark"

    etl_err = ETLError("etl boom", backend="jax", model_name="spider")
    assert etl_err.operation == "etl"

    chat_err = ChatError("chat boom", backend="keras", model_name="gemma-4-E2B")
    assert chat_err.operation == "chat"

    eval_err = EvaluationError("eval boom", backend="jax", model_name="gemma-4-E2B")
    assert eval_err.operation == "evaluate"

    peft_err = PEFTError("peft boom", backend="jax", model_name="gemma-4-E2B")
    assert peft_err.operation == "peft"


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
    assert is_failed_status({"status": "failed: err", "status_code": "completed"}) is True
    assert is_failed_status({"status": ExecutionStatus.ERROR}) is True
    assert is_failed_status({"status": "completed", "error": "boom"}) is True
    assert is_failed_status({"status": "completed", "status_code": ExecutionStatus.FAILED}) is True
    assert is_failed_status({"status": "completed", "status_code": "completed", "error_message": "boom"}) is True
    assert is_failed_status({"status": "completed"}) is False
    assert is_failed_status({"status": 123}) is False
    assert is_failed_status(123) is False


def test_resolve_raise_on_error() -> None:
    """Verify resolve_raise_on_error priority across explicit flag, kwargs, config field, and extra_kwargs."""
    from gemma_4_sql.type_hints import TrainingConfig

    assert resolve_raise_on_error() is False
    assert resolve_raise_on_error(raise_on_error=True) is True
    assert resolve_raise_on_error(raise_on_error=False, kwargs={"raise_on_error": True}) is False
    assert resolve_raise_on_error(kwargs={"raise_on_error": True}) is True
    assert resolve_raise_on_error(kwargs={"other": True}) is False

    cfg_true = TrainingConfig(raise_on_error=True)
    assert resolve_raise_on_error(cfg_true) is True

    cfg_extra = TrainingConfig(extra_kwargs={"raise_on_error": True})
    assert resolve_raise_on_error(cfg_extra) is True

    cfg_false = TrainingConfig(raise_on_error=False, extra_kwargs={"other": 1})
    assert resolve_raise_on_error(cfg_false) is False

    class _OnlyExtraConfig:
        """Config carrying only extra_kwargs without raise_on_error property."""

        extra_kwargs: dict[str, str | int | float | bool | None] | None = None

    assert resolve_raise_on_error(_OnlyExtraConfig()) is False
    assert resolve_raise_on_error("plain_string") is False
    assert is_failed_status(ExecutionStatus.BACKEND_UNAVAILABLE) is True
    assert is_failed_status(ExecutionStatus.LOADED) is False

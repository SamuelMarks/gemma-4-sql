"""Common Direct Preference Optimization (DPO) loss and epoch loop utilities."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Protocol, Self, SupportsFloat, overload, runtime_checkable

from gemma_4_sql._types_tensors import ScalarItemExtractorProtocol, extract_scalar_float
from gemma_4_sql.exceptions import (
    OPERATIONAL_EXCEPTIONS,
    ConfigurationError,
    DependencyMissingError,
    ExecutionStatus,
    StatusValue,
    TrainingError,
    is_failed_status,
    resolve_raise_on_error,
)
from gemma_4_sql.type_hints import DPOConfig, DPOLoopState, DPOResultDict

logger = logging.getLogger(__name__)


@runtime_checkable
class ArithmeticTensorProtocol(SupportsFloat, Protocol):
    """Protocol for numeric or tensor values supporting DPO arithmetic (`-`, `*`, unary `-`)."""

    def __sub__(self, other: Self | float, /) -> Self:
        """Subtract another tensor or scalar."""
        ...

    def __rsub__(self, other: Self | float, /) -> Self:
        """Right-subtract this tensor from a scalar or tensor."""
        ...

    def __mul__(self, other: Self | float, /) -> Self:
        """Multiply by a scalar or tensor."""
        ...

    def __rmul__(self, other: Self | float, /) -> Self:
        """Right-multiply by a scalar or tensor."""
        ...

    def __neg__(self) -> Self:
        """Negate elementwise."""
        ...


@runtime_checkable
class DetachableTensorProtocol(Protocol):
    """Protocol for tensors supporting a `.detach()` method (such as PyTorch tensors)."""

    def detach(self) -> Self:
        """Return a tensor detached from the autograd computation graph."""
        ...


@runtime_checkable
class MeanReducibleTensorProtocol(Protocol):
    """Protocol for tensors supporting `.mean()` reduction."""

    def mean(self) -> Self | float:
        """Compute the arithmetic mean of tensor elements."""
        ...


ScalarItemProtocol = ScalarItemExtractorProtocol


def run_dpo_wrapper(
    backend_name: str,
    config: DPOConfig,
    execute_fn: Callable[[str, str, float, int, float, int], tuple[str, float]],
    **kwargs: StatusValue,
) -> DPOResultDict:
    """Execute a backend DPO loop with standardized config unpacking, error handling, and result formatting.

    Args:
        backend_name: Target backend identifier ('jax', 'keras', 'maxtext', 'mlx', 'pytorch').
        config: `DPOConfig` dataclass instance.
        execute_fn: Backend callable `(model_name, dataset, beta, epochs, lr, batch_size) -> (status, final_loss)`.
        **kwargs: Runtime keyword overrides (including `raise_on_error`).

    Returns:
        Structured `DPOResultDict` containing DPO execution status and final loss.

    Raises:
        DependencyMissingError: If required backend DPO dependencies are missing.
        ConfigurationError: If DPO configuration parameters are invalid.
        TrainingError: If `execute_fn` raises an exception and `raise_on_error` is True.

    """
    model_name = str(config.model_name)
    dataset = str(config.dataset)
    beta = float(config.beta)
    epochs = int(config.epochs)
    learning_rate = float(config.learning_rate)
    batch_size = int(config.batch_size)
    raise_on_error = resolve_raise_on_error(config, kwargs=dict(kwargs))

    final_loss = 0.0
    error_message: str | None = None
    try:
        status, final_loss = execute_fn(model_name, dataset, beta, epochs, learning_rate, batch_size)
        status_code = ExecutionStatus.FAILED.value if is_failed_status(status) else ExecutionStatus.COMPLETED.value
    except (DependencyMissingError, ConfigurationError):
        raise
    except OPERATIONAL_EXCEPTIONS as e:
        logger.exception("%s DPO failed: ", backend_name)
        if raise_on_error:
            raise TrainingError(
                f"{backend_name} DPO failed: {e!s}",
                backend=backend_name,
                model_name=model_name,
                operation="dpo",
                cause=e,
            ) from e
        error_message = str(e)
        status = f"failed: {error_message}"
        status_code = ExecutionStatus.FAILED.value

    is_synthetic = dataset.strip().lower() in {"synthetic", "synthetic-dpo", "dummy"}
    result: DPOResultDict = {
        "backend": backend_name,
        "action": "dpo",
        "model": model_name,
        "dataset": dataset,
        "beta": beta,
        "status": status,
        "status_code": status_code,
        "final_loss": float(final_loss),
        "dataset_source": str(kwargs.get("dataset_source") or ("synthetic_fallback" if is_synthetic else "loaded")),
    }
    if error_message is not None:
        result["error_message"] = error_message
    return result


def _detach_if_supported[TensorT: ArithmeticTensorProtocol | float](val: TensorT) -> TensorT:
    """Return `val.detach()` when `val` implements `DetachableTensorProtocol`, else `val`."""
    if isinstance(val, DetachableTensorProtocol):
        res: TensorT = val.detach()
        return res
    return val


@overload
def _sub_dpo[TensorT: ArithmeticTensorProtocol | float](a: TensorT, b: TensorT) -> TensorT:
    """Compute `a - b` for matching tensor or float operands."""


@overload
def _sub_dpo[TensorT: ArithmeticTensorProtocol | float](a: TensorT, b: TensorT, *, scale: float) -> TensorT:
    """Compute `(a - b) * scale` for matching tensor or float operands."""


def _sub_dpo(
    a: ArithmeticTensorProtocol | float,
    b: ArithmeticTensorProtocol | float,
    *,
    scale: float = 1.0,
) -> ArithmeticTensorProtocol | float:
    """Compute `(a - b) * scale` for either `ArithmeticTensorProtocol` tensors or `float` scalars.

    Args:
        a: Left-hand tensor or float operand.
        b: Right-hand tensor or float operand.
        scale: Multiplicative scale factor (defaults to `1.0`).

    Returns:
        Scaled difference `(a - b) * scale`.

    """
    if isinstance(a, (int, float)):
        diff_f = float(a) - float(b)
        return diff_f * scale if scale != 1.0 else diff_f
    diff_t = a - b
    return diff_t * scale if scale != 1.0 else diff_t


@overload
def _mul_dpo[TensorT: ArithmeticTensorProtocol | float](beta: float, x: TensorT) -> TensorT:
    """Compute `beta * x` for tensor or float operands."""


@overload
def _mul_dpo[TensorT: ArithmeticTensorProtocol | float](beta: float, x: TensorT, *, scale: float) -> TensorT:
    """Compute `beta * x * scale` for tensor or float operands."""


def _mul_dpo(
    beta: float,
    x: ArithmeticTensorProtocol | float,
    *,
    scale: float = 1.0,
) -> ArithmeticTensorProtocol | float:
    """Compute `beta * x * scale` for either `ArithmeticTensorProtocol` tensors or `float` scalars.

    Args:
        beta: Scalar scaling factor.
        x: Tensor or float operand.
        scale: Optional secondary multiplicative scale factor (defaults to `1.0`).

    Returns:
        Scaled value `beta * x * scale`.

    """
    eff_scale = beta * scale if scale != 1.0 else beta
    if isinstance(x, (int, float)):
        return eff_scale * float(x)
    return eff_scale * x


def generic_dpo_loss[TensorT: ArithmeticTensorProtocol | float](
    policy_chosen_logps: TensorT,
    policy_rejected_logps: TensorT,
    ref_chosen_logps: TensorT,
    ref_rejected_logps: TensorT,
    beta: float,
    log_sigmoid_fn: Callable[..., ArithmeticTensorProtocol | float],
) -> tuple[TensorT | ArithmeticTensorProtocol | float, TensorT, TensorT]:
    """Compute the Direct Preference Optimization (DPO) loss generically across tensor frameworks.

    Args:
        policy_chosen_logps: Log probabilities of chosen completions under the policy model.
        policy_rejected_logps: Log probabilities of rejected completions under the policy model.
        ref_chosen_logps: Log probabilities of chosen completions under the reference model.
        ref_rejected_logps: Log probabilities of rejected completions under the reference model.
        beta: Temperature parameter controlling deviation from the reference model.
        log_sigmoid_fn: Backend-specific elementwise log-sigmoid callable.

    Returns:
        A 3-tuple of `(mean_loss, chosen_rewards, rejected_rewards)`.

    """
    pi_logratios: TensorT = _sub_dpo(policy_chosen_logps, policy_rejected_logps)
    ref_logratios: TensorT = _sub_dpo(ref_chosen_logps, ref_rejected_logps)
    logits: TensorT = _sub_dpo(pi_logratios, ref_logratios)
    loss = -log_sigmoid_fn(_mul_dpo(beta, logits))
    diff_chosen: TensorT = _detach_if_supported(_sub_dpo(policy_chosen_logps, ref_chosen_logps))
    chosen_rewards: TensorT = _mul_dpo(beta, diff_chosen)
    diff_rejected: TensorT = _detach_if_supported(_sub_dpo(policy_rejected_logps, ref_rejected_logps))
    rejected_rewards: TensorT = _mul_dpo(beta, diff_rejected)
    mean_loss = loss.mean() if isinstance(loss, MeanReducibleTensorProtocol) else loss
    return (mean_loss, chosen_rewards, rejected_rewards)


def generic_run_training_epochs[ModelT, OptT, BatchT](
    state: DPOLoopState[ModelT, OptT, BatchT],
    step_fn: Callable[..., ScalarItemProtocol | SupportsFloat | float | int],
) -> float:
    """Run DPO training epochs abstracting backend-specific step execution.

    Args:
        state: `DPOLoopState` containing dataloader, policy/reference models, optimizer, and beta.
        step_fn: Backend-specific callable executing a single DPO optimization step.

    Returns:
        The mean training loss from the final epoch.

    """
    dataloader = state.dataloader
    epochs = state.epochs
    policy_model = state.policy_model
    ref_model = state.ref_model
    optimizer = state.optimizer
    beta = state.beta
    final_loss = 0.0
    for _epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        for batch in dataloader:
            loss = step_fn(policy_model, ref_model, optimizer, batch, beta)
            loss_val = extract_scalar_float(loss)
            epoch_loss += loss_val
            num_batches += 1
        final_loss = epoch_loss / max(1, num_batches)
    return final_loss

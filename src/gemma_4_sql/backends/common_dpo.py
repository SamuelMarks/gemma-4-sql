"""Common Direct Preference Optimization (DPO) loss and epoch loop utilities."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Protocol, Self, SupportsFloat, runtime_checkable

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
from gemma_4_sql.type_hints import DPOConfig, DPOLoopState, DPOResultDict, TrainerState

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


@runtime_checkable
class ScalarItemProtocol(Protocol):
    """Protocol for scalar tensors supporting `.item()` extraction."""

    def item(self) -> float | int | bool:
        """Extract a Python scalar value."""
        ...


@runtime_checkable
class JaxNNLogSoftmaxProtocol[ArrayT](Protocol):
    """Protocol for `jax.nn` providing `log_softmax`."""

    def log_softmax(self, x: ArrayT, axis: int = -1) -> ArrayT:
        """Compute log-softmax along `axis`."""
        ...


@runtime_checkable
class JaxNumpyLogpOpsProtocol[ArrayT](Protocol):
    """Protocol for `jax.numpy` array operations used in `gather_jax_sequence_logps`."""

    def expand_dims(self, a: ArrayT, axis: int) -> ArrayT:
        """Expand the shape of an array at `axis`."""
        ...

    def take_along_axis(self, arr: ArrayT, indices: ArrayT, axis: int) -> ArrayT:
        """Take values from the input array by matching 1D index and data slices."""
        ...

    def squeeze(self, a: ArrayT, axis: int) -> ArrayT:
        """Remove single-dimensional entries from the shape of an array."""
        ...

    def sum(self, a: ArrayT, axis: int = -1) -> ArrayT:
        """Sum of array elements over a given axis."""
        ...

    def not_equal(self, x1: ArrayT, x2: int, /) -> ArrayT:
        """Return `(x1 != x2)` elementwise."""
        ...


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

    result: DPOResultDict = {
        "backend": backend_name,
        "action": "dpo",
        "model": model_name,
        "dataset": dataset,
        "beta": beta,
        "status": status,
        "status_code": status_code,
        "final_loss": float(final_loss),
    }
    if error_message is not None:
        result["error_message"] = error_message
    return result


def generic_dpo_loss[TensorT: ArithmeticTensorProtocol | float](
    policy_chosen_logps: TensorT,
    policy_rejected_logps: TensorT,
    ref_chosen_logps: TensorT,
    ref_rejected_logps: TensorT,
    beta: float,
    log_sigmoid_fn: Callable[..., TensorT],
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
    pi_logratios: TensorT = policy_chosen_logps - policy_rejected_logps  # ty: ignore[unsupported-operator]
    ref_logratios: TensorT = ref_chosen_logps - ref_rejected_logps  # ty: ignore[unsupported-operator]
    logits: TensorT = pi_logratios - ref_logratios  # ty: ignore[unsupported-operator]
    loss = -log_sigmoid_fn(beta * logits)  # ty: ignore[unsupported-operator]
    diff_chosen: TensorT = policy_chosen_logps - ref_chosen_logps  # ty: ignore[unsupported-operator]
    chosen_rewards: TensorT = beta * diff_chosen.detach() if isinstance(diff_chosen, DetachableTensorProtocol) else beta * diff_chosen  # ty: ignore[unsupported-operator]
    diff_rejected: TensorT = policy_rejected_logps - ref_rejected_logps  # ty: ignore[unsupported-operator]
    rejected_rewards: TensorT = beta * diff_rejected.detach() if isinstance(diff_rejected, DetachableTensorProtocol) else beta * diff_rejected  # ty: ignore[unsupported-operator]
    mean_loss = loss.mean() if isinstance(loss, MeanReducibleTensorProtocol) else loss
    return (mean_loss, chosen_rewards, rejected_rewards)


def gather_jax_sequence_logps[ArrayT: ArithmeticTensorProtocol](
    jnn_mod: JaxNNLogSoftmaxProtocol[ArrayT],
    jnp_mod: JaxNumpyLogpOpsProtocol[ArrayT],
    logits: ArrayT,
    labels: ArrayT,
) -> ArrayT:
    """Gather and sum non-padding token log-probabilities along the vocabulary dimension in JAX.

    Args:
        jnn_mod: The `jax.nn` module providing `log_softmax`.
        jnp_mod: The `jax.numpy` module providing `expand_dims`, `take_along_axis`, `squeeze`, and `sum`.
        logits: Model output logits tensor of shape `(batch_size, seq_len, vocab_size)`.
        labels: Target token IDs tensor of shape `(batch_size, seq_len)`.

    Returns:
        Summed non-padding token log probabilities per sequence of shape `(batch_size,)`.

    """
    log_probs = jnn_mod.log_softmax(logits, axis=-1)
    labels_expanded = jnp_mod.expand_dims(labels, axis=-1)
    selected_log_probs = jnp_mod.squeeze(jnp_mod.take_along_axis(log_probs, labels_expanded, axis=-1), axis=-1)
    mask = jnp_mod.not_equal(labels, 0)
    weighted: ArrayT = selected_log_probs * mask
    return jnp_mod.sum(weighted, axis=-1)


def generic_run_training_epochs[ModelT, OptT, BatchT, ParamsT](
    state: DPOLoopState[ModelT, OptT, BatchT] | TrainerState[ModelT, ParamsT, OptT, BatchT],
    step_fn: Callable[..., ScalarItemProtocol | SupportsFloat | float | int],
) -> float:
    """Run DPO training epochs abstracting backend-specific step execution.

    Args:
        state: The trainer state containing dataloader, policy/reference models, optimizer, and beta.
        step_fn: Backend-specific callable executing a single DPO optimization step.

    Returns:
        The mean training loss from the final epoch.

    """
    dataloader = state.dataloader
    if dataloader is None:
        return 0.0
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
            loss_val = float(loss.item()) if isinstance(loss, ScalarItemProtocol) else float(loss)
            epoch_loss += loss_val
            num_batches += 1
        final_loss = epoch_loss / max(1, num_batches)
    return final_loss

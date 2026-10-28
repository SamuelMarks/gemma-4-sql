"""MLX-specific DPO (Direct Preference Optimization) logic."""

from __future__ import annotations

import contextlib
import logging
import types
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Protocol, SupportsFloat, runtime_checkable

from gemma_4_sql.backends.common_dpo import ArithmeticTensorProtocol, generic_dpo_loss, run_dpo_wrapper
from gemma_4_sql.backends.common_train import generic_run_training_epochs
from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.backends.mlx.etl import build_dataloader
from gemma_4_sql.exceptions import DependencyMissingError, StatusValue
from gemma_4_sql.type_hints import (
    DPOConfig,
    DPOLoopState,
    DPOResultDict,
    ETLConfig,
    JSONValue,
    MlxArrayProtocol,
    MlxCausalLMProtocol,
    MlxCoreProtocol,
    MlxNNProtocol,
    TensorType,
    TokenizerProtocol,
    TrainerState,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class _MlxOptimizerProtocol[ModelT](Protocol):
    """Protocol for an `mlx.optimizers` optimizer supporting `.update(model, grads)`."""

    def update(
        self,
        model: ModelT,
        grads: dict[str, MlxArrayProtocol],
        /,
    ) -> None:
        """Apply gradients to `model` parameters."""
        ...


type MlxDPOBatchValue = TensorType | JSONValue | MlxArrayProtocol | SupportsFloat | Sequence[int] | Sequence[Sequence[int]] | int | float | str | None
type MlxDPOBatch = Mapping[str, MlxDPOBatchValue]
type MlxDPOModel = Callable[..., MlxArrayProtocol | SupportsFloat | str] | MlxCausalLMProtocol


mlx = optional_import("mlx")
mx = optional_import("mlx.core")
nn = optional_import("mlx.nn")
mx_nn: types.ModuleType | None = nn
optim = optional_import("mlx.optimizers")
_raw_load = optional_attr("mlx_lm", "load")
load: Callable[[str], tuple[MlxCausalLMProtocol, TokenizerProtocol | str | None] | MlxCausalLMProtocol] | None = (  # ty: ignore[invalid-assignment]
    _raw_load if callable(_raw_load) else None
)


def dpo_loss[TensorT: ArithmeticTensorProtocol | float](
    policy_chosen_logps: TensorT,
    policy_rejected_logps: TensorT,
    ref_chosen_logps: TensorT,
    ref_rejected_logps: TensorT,
    beta: float = 0.1,
) -> tuple[TensorT | ArithmeticTensorProtocol | float, TensorT, TensorT]:
    """Compute the Direct Preference Optimization (DPO) loss for MLX arrays.

    Args:
        policy_chosen_logps: Log probabilities of chosen completions from the policy model.
        policy_rejected_logps: Log probabilities of rejected completions from the policy model.
        ref_chosen_logps: Log probabilities of chosen completions from the reference model.
        ref_rejected_logps: Log probabilities of rejected completions from the reference model.
        beta: Temperature parameter controlling deviation from the reference model.

    Returns:
        A 3-tuple of `(mean_loss, chosen_rewards, rejected_rewards)`.

    Raises:
        DependencyMissingError: If MLX dependencies (`mx` or `mx_nn`) are missing.

    """
    if mx is None or mx_nn is None:
        raise DependencyMissingError("MLX DPO dependencies are missing.", backend="mlx")
    return generic_dpo_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        beta,
        mx_nn.log_sigmoid,
    )


def _compute_mlx_logps(
    model: MlxDPOModel,
    inputs: MlxDPOBatchValue,
    labels: MlxDPOBatchValue = None,
) -> MlxArrayProtocol:
    """Compute per-sequence summed token log-probabilities for MLX models.

    Args:
        model: Policy or reference MLX model callable.
        inputs: Input token ID array of shape `(batch_size, seq_len)`.
        labels: Optional target token ID array of shape `(batch_size, seq_len)`.

    Returns:
        Per-sequence log-probability tensor of shape `(batch_size,)`.

    """
    raw_logits = model(inputs)  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]
    target = labels if labels is not None else inputs
    if mx is not None and isinstance(raw_logits, MlxArrayProtocol) and raw_logits.ndim >= 2:
        logits: MlxArrayProtocol = raw_logits
        target_arr: MlxArrayProtocol = (
            target if isinstance(target, MlxArrayProtocol) else mx.array(target)  # pyright: ignore[reportArgumentType]
        )
        if labels is None and target_arr.ndim == 2 and int(target_arr.shape[1]) > 1 and int(logits.shape[1]) > 1:
            logits = logits[:, :-1, :]
            target_arr = target_arr[:, 1:]
        log_probs: MlxArrayProtocol = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        indices: MlxArrayProtocol = mx.expand_dims(target_arr, axis=-1)
        gathered: MlxArrayProtocol = mx.take_along_axis(log_probs, indices, axis=-1)
        squeezed: MlxArrayProtocol = mx.squeeze(gathered, axis=-1)
        mask = target_arr != 0  # pyright: ignore[reportUnnecessaryComparison]
        masked = squeezed * mask  # pyright: ignore[reportOperatorIssue]
        res: MlxArrayProtocol = masked.sum(axis=-1)
        return res
    with contextlib.suppress(AttributeError, TypeError):
        return raw_logits.mean(axis=-1)  # pyright: ignore[reportAttributeAccessIssue,reportCallIssue,reportReturnType]  # ty: ignore[unresolved-attribute]
    return raw_logits  # pyright: ignore[reportReturnType]  # ty: ignore[invalid-return-type]


def _loss_fn(
    policy_model: MlxDPOModel,
    ref_model: MlxDPOModel,
    batch: MlxDPOBatch,
    beta: float,
) -> MlxArrayProtocol | ArithmeticTensorProtocol | float:
    """Evaluate scalar DPO loss for `mlx.nn.value_and_grad` differentiation.

    Args:
        policy_model: Trainable MLX policy model.
        ref_model: Frozen MLX reference model.
        batch: Preference batch mapping containing chosen and rejected inputs/labels.
        beta: KL penalty temperature coefficient.

    Returns:
        Scalar DPO step loss.

    """
    ch_inputs = batch.get("chosen_inputs", batch.get("inputs"))
    re_inputs = batch.get("rejected_inputs", batch.get("inputs"))
    ch_labels = batch.get("chosen_labels", batch.get("targets", ch_inputs))
    re_labels = batch.get("rejected_labels", batch.get("targets", re_inputs))
    pi_ch_logps = _compute_mlx_logps(policy_model, ch_inputs, ch_labels)
    pi_re_logps = _compute_mlx_logps(policy_model, re_inputs, re_labels)
    ref_ch_logps = _compute_mlx_logps(ref_model, ch_inputs, ch_labels)
    ref_re_logps = _compute_mlx_logps(ref_model, re_inputs, re_labels)
    (loss, _, _) = dpo_loss(pi_ch_logps, pi_re_logps, ref_ch_logps, ref_re_logps, beta)
    return loss


def _run_dpo_step[ModelT: MlxDPOModel](
    policy_model: ModelT,
    ref_model: ModelT,
    optimizer: _MlxOptimizerProtocol[ModelT] | str | None,
    batch: MlxDPOBatch,
    beta: float,
) -> MlxArrayProtocol | ArithmeticTensorProtocol | float:
    """Run a single MLX DPO training step using functional `nn.value_and_grad`.

    Args:
        policy_model: Trainable MLX policy model.
        ref_model: Frozen MLX reference model.
        optimizer: MLX optimizer instance supporting `update(model, grads)`.
        batch: Preference batch dictionary.
        beta: KL penalty temperature coefficient.

    Returns:
        Scalar DPO step loss.

    """
    if nn is not None:
        loss_and_grad_fn = nn.value_and_grad(policy_model, _loss_fn)
        (loss, grads) = loss_and_grad_fn(policy_model, ref_model, batch, beta)
        with contextlib.suppress(AttributeError, TypeError):
            optimizer.update(policy_model, grads)  # pyright: ignore[reportOptionalMemberAccess,reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]
        with contextlib.suppress(AttributeError, TypeError):
            mx.eval(policy_model.parameters(), optimizer.state if optimizer is not None else {}, loss)  # pyright: ignore[reportOptionalMemberAccess,reportAttributeAccessIssue,reportFunctionMemberAccess]  # ty: ignore[unresolved-attribute]
        return loss
    return _loss_fn(policy_model, ref_model, batch, beta)


def _run_training_epochs[ModelT, OptT, BatchT: MlxDPOBatch, ParamsT](
    state: DPOLoopState[ModelT, OptT, BatchT] | TrainerState[ModelT, ParamsT, OptT, BatchT],
) -> float:
    """Run MLX DPO training epochs over the preference dataloader.

    Args:
        state: `DPOLoopState` holding models, optimizer, dataloader, and beta.

    Returns:
        Mean batch loss from the final epoch.

    """
    dataloader: Iterable[BatchT] = state.dataloader or ()
    policy_model = state.policy_model
    ref_model = state.ref_model
    optimizer = state.optimizer
    beta = state.beta

    def _step(batch: BatchT) -> float:
        """Process a single preference batch and return scalar float loss."""
        assert policy_model is not None
        assert ref_model is not None
        loss = _run_dpo_step(policy_model, ref_model, optimizer, batch, beta)  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]
        try:
            return float(loss.item())  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]
        except (AttributeError, TypeError, ValueError):
            return float(loss)  # pyright: ignore[reportArgumentType]

    return generic_run_training_epochs(state.epochs, dataloader, _step)


def _execute_dpo(
    model_name: str,
    dataset: str,
    beta: float,
    epochs: int,
    learning_rate: float,
    batch_size: int = 2,
) -> tuple[str, float]:
    """Execute the core DPO loop for MLX.

    Args:
        model_name: Target model name.
        dataset: Dataset name.
        beta: DPO beta temperature.
        epochs: Number of training epochs.
        learning_rate: Optimizer learning rate.
        batch_size: Training batch size.

    Returns:
        Tuple of `(status, final_loss)`.

    Raises:
        ValueError: If dataloader is invalid.

    """
    assert load is not None
    assert optim is not None
    loaded_p = load(model_name)
    policy_model: MlxCausalLMProtocol = loaded_p[0] if isinstance(loaded_p, (tuple, list)) else loaded_p
    loaded_r = load(model_name)
    ref_model: MlxCausalLMProtocol = loaded_r[0] if isinstance(loaded_r, (tuple, list)) else loaded_r
    optimizer: _MlxOptimizerProtocol[MlxCausalLMProtocol] = optim.AdamW(learning_rate=learning_rate)
    data_dict = build_dataloader(ETLConfig(dataset_name=dataset, split="train", batch_size=batch_size))
    dataloader = data_dict.get("loader", None)
    has_loader = isinstance(dataloader, Iterable)
    if not has_loader:
        raise ValueError(f"Invalid dataloader for dataset: {dataset}")
    final_loss = _run_training_epochs(
        DPOLoopState(
            dataloader=dataloader,
            epochs=epochs,
            policy_model=policy_model,
            ref_model=ref_model,
            optimizer=optimizer,
            beta=beta,
        )
    )
    return "completed", float(final_loss)


def run_dpo(config: DPOConfig, **kwargs: StatusValue) -> DPOResultDict:
    """Execute Direct Preference Optimization (DPO) using the MLX backend.

    Args:
        config: DPO configuration parameters.
        **kwargs: Hyperparameters for DPO (e.g., `beta`, `learning_rate`, `raise_on_error`).

    Returns:
        Structured `DPOResultDict` containing backend, model, dataset, beta, status, and final_loss.

    Raises:
        DependencyMissingError: If MLX dependencies are missing.
        TrainingError: If DPO fails and `raise_on_error` is enabled.

    """
    if mlx is None or mx is None or nn is None or optim is None or load is None:
        raise DependencyMissingError("MLX dependencies are missing.", backend="mlx")
    return run_dpo_wrapper(
        "mlx",
        config,
        lambda m, d, b, ep, lr, bs: _execute_dpo(m, d, b, ep, lr, batch_size=bs),
        **kwargs,
    )


__all__ = [
    "MlxArrayProtocol",
    "MlxCausalLMProtocol",
    "MlxCoreProtocol",
    "MlxNNProtocol",
    "dpo_loss",
    "run_dpo",
]

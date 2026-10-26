"""MLX-specific DPO (Direct Preference Optimization) logic."""

from __future__ import annotations

import logging
import types
from collections.abc import Iterable, Mapping
from typing import Any

from gemma_4_sql.backends.common_dpo import generic_dpo_loss, run_dpo_wrapper
from gemma_4_sql.backends.common_train import generic_run_training_epochs
from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.backends.mlx.etl import build_dataloader
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import DPOConfig, DPOLoopState, DPOResultDict, ETLConfig, JSONDict, MlxCausalLMProtocol, TensorType, TrainerState

logger = logging.getLogger(__name__)

mlx = optional_import("mlx")
mx = optional_import("mlx.core")
nn = optional_import("mlx.nn")
mx_nn: types.ModuleType | None = nn
optim = optional_import("mlx.optimizers")
load = optional_attr("mlx_lm", "load")


def dpo_loss(
    policy_chosen_logps: TensorType,
    policy_rejected_logps: TensorType,
    ref_chosen_logps: TensorType,
    ref_rejected_logps: TensorType,
    beta: float = 0.1,
) -> tuple[Any, Any, Any]:
    """Compute the Direct Preference Optimization (DPO) loss for MLX arrays.

    Args:
        policy_chosen_logps: Log probabilities of chosen completions from the policy model.
        policy_rejected_logps: Log probabilities of rejected completions from the policy model.
        ref_chosen_logps: Log probabilities of chosen completions from the reference model.
        ref_rejected_logps: Log probabilities of rejected completions from the reference model.
        beta: Temperature parameter controlling deviation from the reference model.

    Returns:
        A 3-tuple of `(mean_loss, chosen_rewards, rejected_rewards)`.

    """
    if mx is None or mx_nn is None:
        return (0.0, 0.0, 0.0)

    def _fallback_log_sig(x: Any) -> Any:
        """Compute log-sigmoid `-log(1 + exp(-x))` or negative fallback when `nn.log_sigmoid` is absent."""
        if hasattr(mx, "logaddexp"):
            try:
                return -mx.logaddexp(0.0, -x)
            except (TypeError, ValueError):
                pass
        return mx.negative(x) if hasattr(mx, "negative") else -x

    losses_mod = getattr(mx_nn, "losses", None)
    log_sig_fn = getattr(losses_mod, "log_sigmoid", getattr(mx_nn, "log_sigmoid", _fallback_log_sig))
    return generic_dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta, log_sig_fn)


def _compute_mlx_logps(model: MlxCausalLMProtocol, inputs: Any, labels: Any = None) -> Any:
    """Compute per-sequence summed token log-probabilities for MLX models.

    Args:
        model: Policy or reference MLX model callable.
        inputs: Input token ID array of shape `(batch_size, seq_len)`.
        labels: Optional target token ID array of shape `(batch_size, seq_len)`.

    Returns:
        Per-sequence log-probability tensor of shape `(batch_size,)`.

    """
    logits: Any = model(inputs)
    target = labels if labels is not None else inputs
    ndim = getattr(logits, "ndim", 0)
    if mx is not None and isinstance(ndim, int) and ndim >= 2 and hasattr(mx, "take_along_axis") and hasattr(mx, "expand_dims") and target is not None and not isinstance(target, str):
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True) if hasattr(mx, "logsumexp") else logits
        target_arr = target if hasattr(target, "shape") else mx.array(target)
        indices = mx.expand_dims(target_arr, axis=-1)
        gathered = mx.take_along_axis(log_probs, indices, axis=-1)
        squeezed = mx.squeeze(gathered, axis=-1) if hasattr(mx, "squeeze") else gathered
        mask = target_arr != 0
        masked = squeezed * mask if hasattr(squeezed, "__mul__") else squeezed
        return masked.sum(axis=-1) if hasattr(masked, "sum") else masked
    return logits.mean(axis=-1) if hasattr(logits, "mean") else logits


def _loss_fn(policy_model: MlxCausalLMProtocol, ref_model: MlxCausalLMProtocol, batch: Mapping[str, Any], beta: float) -> Any:
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


def _run_dpo_step(policy_model: MlxCausalLMProtocol, ref_model: MlxCausalLMProtocol, optimizer: Any, batch: JSONDict, beta: float) -> Any:
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
    if nn is not None and hasattr(nn, "value_and_grad"):
        loss_and_grad_fn = nn.value_and_grad(policy_model, _loss_fn)
        (loss, grads) = loss_and_grad_fn(policy_model, ref_model, batch, beta)
        if optimizer is not None and hasattr(optimizer, "update"):
            optimizer.update(policy_model, grads)
        if mx is not None and hasattr(mx, "eval") and hasattr(policy_model, "parameters"):
            opt_state = getattr(optimizer, "state", {})
            mx.eval(policy_model.parameters(), opt_state, loss)
        return loss
    return _loss_fn(policy_model, ref_model, batch, beta)


def _run_training_epochs(state: DPOLoopState[Any, Any, Any] | TrainerState[Any, Any, Any, Any]) -> float:
    """Run MLX DPO training epochs over the preference dataloader.

    Args:
        state: `DPOLoopState` holding models, optimizer, dataloader, and beta.

    Returns:
        Mean batch loss from the final epoch.

    """
    dataloader: Iterable[JSONDict] = state.dataloader or ()
    policy_model: Any = state.policy_model
    ref_model: Any = state.ref_model
    optimizer = state.optimizer
    beta = state.beta

    def _step(batch: JSONDict) -> float:
        """Process a single preference batch and return scalar float loss."""
        loss = _run_dpo_step(policy_model, ref_model, optimizer, batch, beta)
        return float(loss.item() if hasattr(loss, "item") else loss)

    return generic_run_training_epochs(state.epochs, dataloader, _step)


def _execute_dpo(model_name: str, dataset: str, beta: float, epochs: int, learning_rate: float, batch_size: int = 2) -> tuple[str, float]:
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
    loaded_p: Any = load(model_name)
    policy_model: MlxCausalLMProtocol = loaded_p[0] if isinstance(loaded_p, (tuple, list)) else loaded_p
    loaded_r: Any = load(model_name)
    ref_model: MlxCausalLMProtocol = loaded_r[0] if isinstance(loaded_r, (tuple, list)) else loaded_r
    optimizer = optim.AdamW(learning_rate=learning_rate)
    data_dict = build_dataloader(ETLConfig(dataset_name=dataset, split="train", batch_size=batch_size))
    dataloader = data_dict.get("loader", None)
    if dataloader is None or not hasattr(dataloader, "__iter__"):
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


def run_dpo(config: DPOConfig, **kwargs: object) -> DPOResultDict:
    """Execute Direct Preference Optimization (DPO) using the MLX backend.

    Args:
        config: DPO configuration parameters.
        **kwargs: Hyperparameters for DPO (e.g., beta, learning_rate, raise_on_error).

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

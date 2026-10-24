"""Common Direct Preference Optimization (DPO) loss and epoch loop utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from gemma_4_sql.type_hints import TensorType, TrainerState


def generic_dpo_loss(
    policy_chosen_logps: TensorType,
    policy_rejected_logps: TensorType,
    ref_chosen_logps: TensorType,
    ref_rejected_logps: TensorType,
    beta: float,
    log_sigmoid_fn: Callable[[TensorType], TensorType],
) -> tuple[TensorType, TensorType, TensorType]:
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
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = pi_logratios - ref_logratios
    loss = -log_sigmoid_fn(beta * logits)
    diff_chosen = policy_chosen_logps - ref_chosen_logps
    detach_ch: Callable[[], TensorType] | None = getattr(diff_chosen, "detach", None)
    chosen_rewards = beta * detach_ch() if detach_ch is not None else beta * diff_chosen
    diff_rejected = policy_rejected_logps - ref_rejected_logps
    detach_re: Callable[[], TensorType] | None = getattr(diff_rejected, "detach", None)
    rejected_rewards = beta * detach_re() if detach_re is not None else beta * diff_rejected
    mean_loss = loss.mean() if hasattr(loss, "mean") else loss
    return (mean_loss, chosen_rewards, rejected_rewards)


def generic_run_training_epochs(
    state: TrainerState[Any, Any, Any, Any],
    step_fn: Callable[[Any, Any, Any, Any, float], Any],
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
            loss_val = float(loss.item() if hasattr(loss, "item") else loss)
            epoch_loss += loss_val
            num_batches += 1
        final_loss = epoch_loss / max(1, num_batches)
    return final_loss

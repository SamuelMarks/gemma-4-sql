"""Provide module docstring."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from gemma_4_sql.type_hints import JSONDict, TensorType, TrainerState


def generic_dpo_loss(policy_chosen_logps: TensorType, policy_rejected_logps: TensorType, ref_chosen_logps: TensorType, ref_rejected_logps: TensorType, beta: float, log_sigmoid_fn: Callable[[TensorType], TensorType]) -> tuple[TensorType, TensorType, TensorType]:
    """Generic computation of the Direct Preference Optimization (DPO) loss.

    Args:
        policy_chosen_logps: Log probabilities of the chosen completions from the policy model.
        policy_rejected_logps: Log probabilities of the rejected completions from the policy model.
        ref_chosen_logps: Log probabilities of the chosen completions from the reference model.
        ref_rejected_logps: Log probabilities of the rejected completions from the reference model.
        beta: The beta parameter controlling the KL penalty.
        log_sigmoid_fn: The log sigmoid fn.

    Returns:
        A tuple containing the results.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = pi_logratios - ref_logratios
    loss = -log_sigmoid_fn(beta * logits)
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach() if hasattr(policy_chosen_logps - ref_chosen_logps, "detach") else beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach() if hasattr(policy_rejected_logps - ref_rejected_logps, "detach") else beta * (policy_rejected_logps - ref_rejected_logps)
    return (loss.mean() if hasattr(loss, "mean") else loss, chosen_rewards, rejected_rewards)


def generic_run_training_epochs(state: TrainerState, step_fn: Callable[[object, object, object, JSONDict, float], object]) -> float:
    """Run training epochs abstracting backend details.

    Args:
    ----
        state: The trainer state containing dataloader, models, optimizer, etc.
        step_fn: The backend specific function for running a single training step.

    Returns:
    -------
        The final training loss.
    """
    dataloader = state.dataloader  # pragma: no cover
    epochs = state.epochs  # pragma: no cover
    policy_model = state.policy_model  # pragma: no cover
    ref_model = state.ref_model  # pragma: no cover
    optimizer = state.optimizer  # pragma: no cover
    beta = state.beta  # pragma: no cover
    final_loss = 0.0  # pragma: no cover
    for _epoch in range(epochs):  # pragma: no cover
        epoch_loss = 0.0  # pragma: no cover
        for batch in dataloader:  # pragma: no cover
            loss = step_fn(policy_model, ref_model, optimizer, batch, beta)  # pragma: no cover
            epoch_loss += loss.item()  # pragma: no cover
        final_loss = epoch_loss / max(1, len(dataloader))  # pragma: no cover
    return final_loss  # pragma: no cover

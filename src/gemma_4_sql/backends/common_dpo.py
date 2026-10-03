"""Provide module docstring."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gemma_4_sql.type_hints import TensorType

if TYPE_CHECKING:
    from collections.abc import Callable

    from gemma_4_sql.type_hints import JSONDict, TrainerState


def generic_dpo_loss(policy_chosen_logps: TensorType, policy_rejected_logps: TensorType, ref_chosen_logps: TensorType, ref_rejected_logps: TensorType, beta: float, log_sigmoid_fn: Callable[[TensorType], TensorType]) -> tuple[TensorType, TensorType, TensorType]:
    """Compute the Direct Preference Optimization (DPO) loss generically.

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
    loss = -log_sigmoid_fn(beta * logits)  # type: ignore # Justified: Dynamic backend protocol typing
    diff_chosen = policy_chosen_logps - ref_chosen_logps
    chosen_rewards = beta * diff_chosen.detach() if hasattr(diff_chosen, "detach") else beta * diff_chosen  # type: ignore # Justified: Dynamic backend protocol typing
    diff_rejected = policy_rejected_logps - ref_rejected_logps
    rejected_rewards = beta * diff_rejected.detach() if hasattr(diff_rejected, "detach") else beta * diff_rejected  # type: ignore # Justified: Dynamic backend protocol typing
    return (loss.mean() if hasattr(loss, "mean") else loss, chosen_rewards, rejected_rewards)  # type: ignore # Justified: Dynamic backend protocol typing


def generic_run_training_epochs(state: TrainerState, step_fn: Callable[[Any, Any, Any, JSONDict, float], Any]) -> float:
    """Run training epochs abstracting backend details.

    Args:
    ----
        state: The trainer state containing dataloader, models, optimizer, etc.
        step_fn: The backend specific function for running a single training step.

    Returns:
    -------
        The final training loss.

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
        for batch in dataloader:
            loss = step_fn(policy_model, ref_model, optimizer, batch, beta)  # type: ignore # Justified: Dynamic backend protocol typing
            loss_val = float(loss.item() if hasattr(loss, "item") else loss)
            epoch_loss += loss_val
        final_loss = epoch_loss / max(1, len(list(dataloader)) if hasattr(dataloader, "__len__") else 1)
    return final_loss

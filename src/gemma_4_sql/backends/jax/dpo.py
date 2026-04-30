"""
JAX-specific DPO (Direct Preference Optimization) logic.
"""

from __future__ import annotations

from typing import Any

try:
    import jax.numpy as jnp
    import jax.nn as jnn  # pragma: no cover
except ImportError:
    jnp = None
    jnn = None

def dpo_loss(
    policy_chosen_logps: Any,
    policy_rejected_logps: Any,
    ref_chosen_logps: Any,
    ref_rejected_logps: Any,
    beta: float = 0.1,
) -> tuple[Any, Any, Any]:
    """
    Computes the DPO loss for JAX.

    Args:
        policy_chosen_logps: Log probabilities of chosen responses from policy model.
        policy_rejected_logps: Log probabilities of rejected responses from policy model.
        ref_chosen_logps: Log probabilities of chosen responses from reference model.
        ref_rejected_logps: Log probabilities of rejected responses from reference model.
        beta: Temperature parameter for the DPO loss.

    Returns:
        A tuple of (loss, chosen_rewards, rejected_rewards).
    """
    if jnp is None or jnn is None:
        return 0.0, 0.0, 0.0

    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = pi_logratios - ref_logratios

    loss = -jnn.log_sigmoid(beta * logits)
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)

    return jnp.mean(loss), jnp.mean(chosen_rewards), jnp.mean(rejected_rewards)

def run_dpo(model_name: str, dataset: str, beta: float = 0.1) -> dict[str, Any]:
    """
    Runs a mocked DPO training loop for JAX.

    Args:
        model_name: The name of the model.
        dataset: The dataset name.
        beta: The beta temperature parameter.

    Returns:
        A dict with the execution status and metrics.
    """
    if jnp is not None:
        status = "completed"
        pi_ch = jnp.array([0.1, 0.2])
        pi_re = jnp.array([-0.1, -0.2])
        ref_ch = jnp.array([0.05, 0.1])
        ref_re = jnp.array([-0.05, -0.1])
        loss, _, _ = dpo_loss(pi_ch, pi_re, ref_ch, ref_re, beta)
        final_loss = loss.item()
    else:
        status = "mocked_missing_jax"
        final_loss = 0.0

    return {
        "backend": "jax",
        "action": "dpo",
        "model": model_name,
        "dataset": dataset,
        "beta": beta,
        "status": status,
        "final_loss": final_loss,
    }

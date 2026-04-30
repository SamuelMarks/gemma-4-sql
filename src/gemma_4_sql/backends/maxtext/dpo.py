"""
MaxText-specific DPO (Direct Preference Optimization) logic.
"""

from __future__ import annotations

from typing import Any

from gemma_4_sql.backends.jax.dpo import dpo_loss as jax_dpo_loss

try:
    import jax.numpy as jnp
except ImportError:
    jnp = None

def dpo_loss(
    policy_chosen_logps: Any,
    policy_rejected_logps: Any,
    ref_chosen_logps: Any,
    ref_rejected_logps: Any,
    beta: float = 0.1,
) -> tuple[Any, Any, Any]:
    """
    Computes the DPO loss for MaxText (using JAX under the hood).

    Args:
        policy_chosen_logps: Log probabilities of chosen responses from policy model.
        policy_rejected_logps: Log probabilities of rejected responses from policy model.
        ref_chosen_logps: Log probabilities of chosen responses from reference model.
        ref_rejected_logps: Log probabilities of rejected responses from reference model.
        beta: Temperature parameter for the DPO loss.

    Returns:
        A tuple of (loss, chosen_rewards, rejected_rewards).
    """
    return jax_dpo_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        beta,
    )

def run_dpo(model_name: str, dataset: str, beta: float = 0.1) -> dict[str, Any]:
    """
    Runs a mocked DPO training loop for MaxText.

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
        final_loss = float(loss.item())
    else:
        status = "mocked_missing_maxtext"
        final_loss = 0.0

    return {
        "backend": "maxtext",
        "action": "dpo",
        "model": model_name,
        "dataset": dataset,
        "beta": beta,
        "status": status,
        "final_loss": final_loss,
    }

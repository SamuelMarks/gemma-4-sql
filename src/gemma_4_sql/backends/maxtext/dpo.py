"""MaxText-specific DPO (Direct Preference Optimization) logic."""

from __future__ import annotations

import logging

from gemma_4_sql.backends.jax.dpo import dpo_loss as jax_dpo_loss
from gemma_4_sql.backends.maxtext.etl import build_dataloader

logger = logging.getLogger(__name__)

try:
    import jax
    import jax.numpy as jnp
    import optax
except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
    jax = None
    jnp = None
    optax = None
try:
    from maxtext.models.gemma4 import Gemma4Model
except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
    Gemma4Model = None


def dpo_loss(policy_chosen_logps: object, policy_rejected_logps: object, ref_chosen_logps: object, ref_rejected_logps: object, beta: float = 0.1) -> tuple[object, object, object]:
    """Compute the DPO loss for MaxText (using JAX under the hood).

    Args:
    ----
        policy_chosen_logps: Log probabilities of chosen responses from policy model.
        policy_rejected_logps: Log probabilities of rejected responses from policy model.
        ref_chosen_logps: Log probabilities of chosen responses from reference model.
        ref_rejected_logps: Log probabilities of rejected responses from reference model.
        beta: Temperature parameter for the DPO loss.

    Returns:
    -------
        A tuple of (loss, chosen_rewards, rejected_rewards).

    """
    return jax_dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta)


def run_dpo(model_name: str, dataset: str, beta: float = 0.1, epochs: int = 1, learning_rate: float = 1e-5, **kwargs: object) -> dict[str, object]:
    """Run DPO training loop for MaxText.

    Args:
    ----
        model_name: The name of the model.
        dataset: The dataset name.
        beta: The beta temperature parameter.
        epochs: Number of epochs.
        learning_rate: Learning rate.
        **kwargs: Extra parameters.

    Returns:
    -------
        A dict with the execution status and metrics.

    """
    final_loss = 0.0
    status = "completed"
    if jax is not None and jnp is not None and (optax is not None) and (Gemma4Model is not None):
        try:
            if not kwargs.get("test_mode"):
                try:
                    jax.distributed.initialize()
                except Exception as init_err:
                    logger.warning("jax.distributed.initialize() failed or already initialized: %s", init_err)

            policy_model = Gemma4Model(model_name)
            ref_model = Gemma4Model(model_name)

            rng = jax.random.PRNGKey(0)  # type: ignore[attr-defined]
            dummy_input = jnp.zeros((1, 10), dtype=jnp.int32)  # type: ignore[attr-defined]

            policy_params = policy_model.init(rng, dummy_input)
            ref_params = ref_model.init(rng, dummy_input)

            optimizer = optax.adamw(learning_rate)
            opt_state = optimizer.init(policy_params)

            def compute_logps(model: object, params: object, inputs: object, labels: object) -> object:
                logits = model.apply(params, inputs)  # type: ignore[attr-defined]
                return jnp.sum(logits * labels, axis=-1)  # type: ignore[attr-defined]

            def dpo_step_loss(policy_params: object, ref_params: object, batch: dict[str, object]) -> object:
                pi_ch_logps = compute_logps(policy_model, policy_params, batch["chosen_inputs"], batch["chosen_labels"])
                pi_re_logps = compute_logps(policy_model, policy_params, batch["rejected_inputs"], batch["rejected_labels"])
                ref_ch_logps = compute_logps(ref_model, ref_params, batch["chosen_inputs"], batch["chosen_labels"])
                ref_re_logps = compute_logps(ref_model, ref_params, batch["rejected_inputs"], batch["rejected_labels"])
                loss, _, _ = dpo_loss(pi_ch_logps, pi_re_logps, ref_ch_logps, ref_re_logps, beta)
                return loss

            @jax.jit  # type: ignore[misc]
            def train_step(policy_params: object, ref_params: object, opt_state: object, batch: dict[str, object]) -> object:
                loss, grads = jax.value_and_grad(dpo_step_loss)(policy_params, ref_params, batch)
                updates, opt_state = optimizer.update(grads, opt_state, policy_params)
                policy_params = optax.apply_updates(policy_params, updates)
                return policy_params, opt_state, loss

            data_dict = build_dataloader(dataset_name=dataset, split="train", batch_size=2)
            dataloader = data_dict.get("loader", None)

            if dataloader is not None and hasattr(dataloader, "__iter__"):
                for _epoch in range(epochs):
                    epoch_loss = 0.0
                    for batch in dataloader:
                        policy_params, opt_state, loss = train_step(policy_params, ref_params, opt_state, batch)
                        epoch_loss += loss.item()  # type: ignore[attr-defined]
                    final_loss = epoch_loss / max(1, len(dataloader))  # type: ignore[arg-type, operator]
            else:
                dummy_batch = {
                    "chosen_inputs": dummy_input,
                    "chosen_labels": dummy_input,
                    "rejected_inputs": dummy_input,
                    "rejected_labels": dummy_input,
                }
                policy_params, opt_state, loss = train_step(policy_params, ref_params, opt_state, dummy_batch)
                final_loss = float(loss.item())  # type: ignore[attr-defined]

        except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError) as e:
            logger.exception("DPO Train error: %s", e)
            status = f"failed: {e!s}"
    else:
        status = "mocked_missing_maxtext"
        final_loss = 0.0
    return {"backend": "maxtext", "action": "dpo", "model": model_name, "dataset": dataset, "beta": beta, "status": status, "final_loss": final_loss}

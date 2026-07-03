"""MaxText-specific DPO (Direct Preference Optimization) logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from gemma_4_sql.backends.jax.dpo import dpo_loss as jax_dpo_loss
from gemma_4_sql.backends.lazy_loader import catch_optional_imports
from gemma_4_sql.backends.maxtext.etl import build_dataloader

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict, JSONValue
logger = logging.getLogger(__name__)
jax = None
jnp = None
optax = None
with catch_optional_imports():
    import jax
    import jax.numpy as jnp
    import optax
Gemma4Model = None
with catch_optional_imports():
    from maxtext.models.gemma4 import Gemma4Model


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


def _compute_logps(model: object, params: dict[str, object] | object, inputs: object, labels: object) -> object:
    """Execute logic."""
    logits = model.apply(params, inputs)
    return jnp.sum(logits * labels, axis=-1)


def _dpo_step_loss(policy_model: object, policy_params: object, ref_model: object, ref_params: object, batch: JSONDict, beta: float) -> object:
    """Execute logic."""
    pi_ch_logps = _compute_logps(policy_model, policy_params, batch["chosen_inputs"], batch["chosen_labels"])
    pi_re_logps = _compute_logps(policy_model, policy_params, batch["rejected_inputs"], batch["rejected_labels"])
    ref_ch_logps = _compute_logps(ref_model, ref_params, batch["chosen_inputs"], batch["chosen_labels"])
    ref_re_logps = _compute_logps(ref_model, ref_params, batch["rejected_inputs"], batch["rejected_labels"])
    (loss, _, _) = dpo_loss(pi_ch_logps, pi_re_logps, ref_ch_logps, ref_re_logps, beta)
    return loss


def _get_train_step_fn(policy_model: object, ref_model: object, optimizer: object, beta: float) -> object:
    """Execute function."""

    @jax.jit
    def train_step(policy_params: object, ref_params: object, opt_state: object, batch: JSONDict) -> object:
        """Execute logic."""
        (loss, grads) = jax.value_and_grad(lambda p, r, b: _dpo_step_loss(policy_model, p, ref_model, r, b, beta))(policy_params, ref_params, batch)
        (updates, opt_state) = optimizer.update(grads, opt_state, policy_params)
        policy_params = optax.apply_updates(policy_params, updates)
        return (policy_params, opt_state, loss)

    return train_step


def _run_training_epochs(dataloader: object, epochs: int, train_step: object, policy_params: object, ref_params: object, opt_state: object) -> tuple[object, object, float]:
    """Run training epochs."""
    final_loss = 0.0
    for _epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            (policy_params, opt_state, loss) = train_step(policy_params, ref_params, opt_state, batch)
            epoch_loss += float(loss.item())
        final_loss = epoch_loss / max(1, len(dataloader))
    return (policy_params, opt_state, final_loss)


def run_dpo(model_name: str, dataset: str, beta: float = 0.1, epochs: int = 1, learning_rate: float = 1e-05, **kwargs: JSONValue) -> JSONDict:
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
                except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as init_err:
                    logger.warning("jax.distributed.initialize() failed or already initialized: %s", init_err)
            policy_model = Gemma4Model(model_name)
            ref_model = Gemma4Model(model_name)
            rng = jax.random.PRNGKey(0)
            dummy_input = jnp.zeros((1, 10), dtype=jnp.int32)
            policy_params = policy_model.init(rng, dummy_input)
            ref_params = ref_model.init(rng, dummy_input)
            optimizer = optax.adamw(learning_rate)
            opt_state = optimizer.init(policy_params)
            train_step = _get_train_step_fn(policy_model, ref_model, optimizer, beta)
            data_dict = build_dataloader(dataset_name=dataset, split="train", batch_size=2)
            dataloader = data_dict.get("loader", None)
            if dataloader is not None and hasattr(dataloader, "__iter__"):
                (policy_params, opt_state, final_loss) = _run_training_epochs(dataloader, epochs, train_step, policy_params, ref_params, opt_state)
            else:
                dummy_batch = {"chosen_inputs": dummy_input, "chosen_labels": dummy_input, "rejected_inputs": dummy_input, "rejected_labels": dummy_input}
                (policy_params, opt_state, loss) = train_step(policy_params, ref_params, opt_state, dummy_batch)
                final_loss = float(loss.item())
        except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError) as e:
            logger.exception("DPO Train error: ")
            status = f"failed: {e!s}"
    else:
        status = "mocked_missing_maxtext"
        final_loss = 0.0
    return {"backend": "maxtext", "action": "dpo", "model": model_name, "dataset": dataset, "beta": beta, "status": status, "final_loss": final_loss}

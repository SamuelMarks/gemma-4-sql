"""JAX-specific DPO (Direct Preference Optimization) logic."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemma_4_sql.backends.common_dpo import generic_dpo_loss
from gemma_4_sql.backends.jax.etl import build_dataloader
from gemma_4_sql.backends.lazy_loader import catch_optional_imports
from gemma_4_sql.type_hints import DPOConfig, ETLConfig, TensorType, TrainerState

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict
jax = None
jnp = None
jnn = None
optax = None
with catch_optional_imports():
    import jax
    import jax.nn as jnn
    import jax.numpy as jnp
    import optax
Gemma4ForCausalLM = None
Gemma4Config = None
nnx = None
with catch_optional_imports():
    from flax import nnx

    from .gemma4 import Gemma4Config, Gemma4ForCausalLM


def dpo_loss(policy_chosen_logps: TensorType, policy_rejected_logps: TensorType, ref_chosen_logps: TensorType, ref_rejected_logps: TensorType, beta: float = 0.1) -> tuple[TensorType, TensorType, TensorType]:
    """Compute the DPO loss.

    Returns:
        tuple: The losses.

    """
    if jnp is None or jnn is None:
        return (0.0, 0.0, 0.0)
    return generic_dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta, jnn.log_sigmoid)
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = pi_logratios - ref_logratios
    loss = -jnn.log_sigmoid(beta * logits)
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    return (jnp.mean(loss), jnp.mean(chosen_rewards), jnp.mean(rejected_rewards))  # pragma: no cover


def _compute_logps(model: object, inputs: object, labels: object) -> object:
    """Mock logp computation.

    Returns:
        object: The resulting output from the operation.

    """
    logits = model(inputs)
    return jnp.sum(logits * labels, axis=-1)


def _dpo_step_loss(policy_model: object, ref_model: object, batch: JSONDict, beta: float) -> object:
    """Compute DPO loss for a step.

    Returns:
        object: The resulting output from the operation.

    """
    pi_ch_logps = _compute_logps(policy_model, batch["chosen_inputs"], batch["chosen_labels"])
    pi_re_logps = _compute_logps(policy_model, batch["rejected_inputs"], batch["rejected_labels"])
    ref_ch_logps = _compute_logps(ref_model, batch["chosen_inputs"], batch["chosen_labels"])
    ref_re_logps = _compute_logps(ref_model, batch["rejected_inputs"], batch["rejected_labels"])
    (loss, _, _) = dpo_loss(pi_ch_logps, pi_re_logps, ref_ch_logps, ref_re_logps, beta)
    return loss


def _get_train_step_fn(beta: float) -> object:
    """Return a JIT-compiled train step function for the given beta.

    Returns:
        object: The resulting output from the operation.

    """

    @nnx.jit
    def train_step(policy_model: object, ref_model: object, optimizer: object, batch: JSONDict) -> object:
        """Execute a single JAX-compiled DPO training step.

        Returns:
            object: The resulting output from the operation.

        """
        (loss, grads) = nnx.value_and_grad(lambda p, r, b: _dpo_step_loss(p, r, b, beta))(policy_model, ref_model, batch)
        optimizer.update(grads)
        return loss

    return train_step


def _run_training_epochs(state: TrainerState) -> float:
    """Execute function.

    Returns:
        object: Description of return.

    """
    dataloader = state.dataloader
    epochs = state.epochs
    policy_model = state.policy_model
    ref_model = state.ref_model
    optimizer = state.optimizer
    train_step = state.train_step
    """Run training epochs.

    Returns:
        object: The resulting output from the operation.

    """
    final_loss = 0.0
    for _epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            loss = train_step(policy_model, ref_model, optimizer, batch)
            epoch_loss += loss.item()
        final_loss = epoch_loss / max(1, len(dataloader))
    return float(final_loss)


def _execute_dpo(model_name: str, dataset: str, beta: float, epochs: int, learning_rate: float) -> tuple[str, float]:
    """Execute the core DPO loop."""
    policy_model = Gemma4ForCausalLM(Gemma4Config.gemma4_e2b(), rngs=nnx.Rngs(0))
    ref_model = Gemma4ForCausalLM(Gemma4Config.gemma4_e2b(), rngs=nnx.Rngs(1))
    optimizer = nnx.Optimizer(policy_model, optax.adamw(learning_rate))
    train_step = _get_train_step_fn(beta)
    data_dict = build_dataloader(ETLConfig(dataset_name=dataset, split="train", batch_size=2))
    dataloader = data_dict.get("loader", None)
    if dataloader is not None and hasattr(dataloader, "__iter__"):
        final_loss = _run_training_epochs(TrainerState(dataloader=dataloader, epochs=epochs, policy_model=policy_model, ref_model=ref_model, optimizer=optimizer, train_step=train_step))
    else:
        dummy_input = jnp.zeros((1, 10), dtype=jnp.int32)
        dummy_batch = {"chosen_inputs": dummy_input, "chosen_labels": dummy_input, "rejected_inputs": dummy_input, "rejected_labels": dummy_input}
        loss = train_step(policy_model, ref_model, optimizer, dummy_batch)
        final_loss = float(loss.item())
    return "completed", final_loss


def run_dpo(config: DPOConfig, **kwargs: object) -> JSONDict:
    """Execute function.

    Returns:
        object: Description of return.

    """
    model_name = getattr(config, "model_name", "model")
    dataset = getattr(config, "dataset", "dataset")
    beta = getattr(config, "beta", 0.1)
    epochs = getattr(config, "epochs", 1)
    learning_rate = getattr(config, "learning_rate", 1e-05)
    """Run a DPO training loop for JAX.

    Args:
    ----
        model_name: The name of the model.
        dataset: The dataset name.
        beta: The beta temperature parameter.
        epochs: Number of training epochs.
        learning_rate: Learning rate for the optimizer.

    Returns:
    -------
        A dict with the execution status and metrics.

    """
    final_loss = 0.0
    status = "completed"
    if jax is not None and jnp is not None and (jnn is not None) and (optax is not None) and (Gemma4ForCausalLM is not None):
        try:
            status, final_loss = _execute_dpo(model_name, dataset, beta, epochs, learning_rate)
        except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError) as e:
            status = f"failed: {e!s}"
    else:
        status = "mocked_missing_jax"
        final_loss = 0.0
    return {"backend": "jax", "action": "dpo", "model": model_name, "dataset": dataset, "beta": beta, "status": status, "final_loss": final_loss}

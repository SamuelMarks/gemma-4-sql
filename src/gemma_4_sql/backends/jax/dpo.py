"""JAX-specific DPO (Direct Preference Optimization) logic."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gemma_4_sql.backends.common_dpo import generic_dpo_loss
from gemma_4_sql.backends.jax.etl import build_dataloader
from gemma_4_sql.type_hints import DPOConfig, ETLConfig, TrainerState

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict

try:
    import jax as _jax
    import jax.nn as _jnn
    import jax.numpy as _jnp
    import optax as _optax
    from flax import nnx as _nnx

    from .gemma4 import Gemma4Config as _Gemma4Config
    from .gemma4 import Gemma4ForCausalLM as _Gemma4ForCausalLM

    jax: Any = _jax
    jnn: Any = _jnn
    jnp: Any = _jnp
    optax: Any = _optax
    nnx: Any = _nnx
    Gemma4Config: Any = _Gemma4Config
    Gemma4ForCausalLM: Any = _Gemma4ForCausalLM
except (ImportError, AttributeError):
    jax = None
    jnn = None
    jnp = None
    optax = None
    nnx = None
    Gemma4Config = None
    Gemma4ForCausalLM = None


def dpo_loss(policy_chosen_logps: Any, policy_rejected_logps: Any, ref_chosen_logps: Any, ref_rejected_logps: Any, beta: float = 0.1) -> tuple[Any, Any, Any]:
    """Compute the DPO loss.

    Args:
        policy_chosen_logps: Log probabilities of the chosen completions from the policy model.
        policy_rejected_logps: Log probabilities of the rejected completions from the policy model.
        ref_chosen_logps: Log probabilities of the chosen completions from the reference model.
        ref_rejected_logps: Log probabilities of the rejected completions from the reference model.
        beta: The beta parameter controlling the KL penalty.

    Returns:
        A tuple containing the results.
    """
    if jnp is None or jnn is None:
        return (0.0, 0.0, 0.0)
    return generic_dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta, jnn.log_sigmoid)


def _compute_logps(model: Any, inputs: Any, labels: Any) -> Any:
    """Compute exact log probabilities for DPO math using categorical cross-entropy approach.

    Returns:
        The resulting output from the operation.
    """
    logits = model(inputs)
    # The labels act as the vocabulary indices of the correct next token.
    # Compute log_softmax over the vocabulary dimension (usually axis -1)
    log_probs = jnn.log_softmax(logits, axis=-1)

    # Gather the log probability of the true next token (label).
    # Since jax operations need to be jitted, we use take_along_axis
    # labels shape is (batch_size, sequence_length)
    # log_probs shape is (batch_size, sequence_length, vocab_size)
    labels_expanded = jnp.expand_dims(labels, axis=-1)
    selected_log_probs = jnp.take_along_axis(log_probs, labels_expanded, axis=-1)

    # Remove the extra dimension and sum over the sequence length
    selected_log_probs = jnp.squeeze(selected_log_probs, axis=-1)
    # We might want to mask out padding tokens in the future, assuming non-zero labels are valid tokens for now
    mask = labels != 0
    return jnp.sum(selected_log_probs * mask, axis=-1)


def _dpo_step_loss(policy_model: Any, ref_model: Any, batch: JSONDict, beta: float) -> Any:
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

    def train_step(policy_model: Any, ref_model: Any, optimizer: Any, batch: JSONDict) -> Any:
        """Execute a single JAX-compiled DPO training step.

        Returns:
            object: The resulting output from the operation.

        """
        if nnx is not None and hasattr(nnx, "value_and_grad"):
            (loss, grads) = nnx.value_and_grad(lambda p, r, b: _dpo_step_loss(p, r, b, beta))(policy_model, ref_model, batch)
        else:
            loss, grads = 0.0, None
        if optimizer is not None and hasattr(optimizer, "update"):
            optimizer.update(grads)
        return loss

    if nnx is not None and hasattr(nnx, "jit"):
        return nnx.jit(train_step)
    return train_step


def _run_training_epochs(state: TrainerState) -> float:
    """Execute function.

    Returns:
        The execution result.

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
            loss_val = float(loss.item() if hasattr(loss, "item") else loss)
            epoch_loss += loss_val
        final_loss = epoch_loss / max(1, len(dataloader))
    return float(final_loss)


def _execute_dpo(model_name: str, dataset: str, beta: float, epochs: int, learning_rate: float, batch_size: int = 2) -> tuple[str, float]:
    """Execute the core DPO loop.

    Args:
        model_name: The name of the model.
        dataset: The dataset name.
        beta: The beta temperature parameter.
        epochs: Number of training epochs.
        learning_rate: Learning rate for the optimizer.
        batch_size: Batch size for training.

    Returns:
        A tuple of (status, final_loss).

    Raises:
        DependencyMissingError: If JAX dependencies are missing.
        ValueError: If dataloader is invalid.
    """
    if jax is None or jnp is None or optax is None or Gemma4ForCausalLM is None or nnx is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("JAX dependencies are missing for DPO.")

    policy_model = Gemma4ForCausalLM(Gemma4Config.gemma4_e2b(), rngs=nnx.Rngs(0))
    ref_model = Gemma4ForCausalLM(Gemma4Config.gemma4_e2b(), rngs=nnx.Rngs(1))
    optimizer = nnx.Optimizer(policy_model, optax.adamw(learning_rate))
    train_step = _get_train_step_fn(beta)
    data_dict = build_dataloader(ETLConfig(dataset_name=dataset, split="train", batch_size=batch_size))
    dataloader = data_dict.get("loader", None)

    if dataloader is None or not hasattr(dataloader, "__iter__"):
        raise ValueError(f"Invalid dataloader for dataset: {dataset}")

    final_loss = _run_training_epochs(TrainerState(dataloader=dataloader, epochs=epochs, policy_model=policy_model, ref_model=ref_model, optimizer=optimizer, train_step=train_step))
    return "completed", final_loss


def run_dpo(config: DPOConfig, **kwargs: object) -> JSONDict:
    """Run a DPO training loop for JAX.

    Args:
        config: DPO training configuration.
        **kwargs: Hyperparameters for DPO (e.g., beta, learning_rate).

    Returns:
        A dict with the execution status and metrics.

    Raises:
        DependencyMissingError: If JAX dependencies are missing.
    """
    model_name = getattr(config, "model_name", "model")
    dataset = getattr(config, "dataset", "dataset")
    beta = getattr(config, "beta", 0.1)
    epochs = getattr(config, "epochs", 1)
    learning_rate = getattr(config, "learning_rate", 1e-05)
    batch_size = getattr(config, "batch_size", 2)
    final_loss = 0.0
    status = "completed"
    if jax is None or jnp is None or jnn is None or optax is None or Gemma4ForCausalLM is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("JAX DPO dependencies are missing.")

    try:
        status, final_loss = _execute_dpo(model_name, dataset, beta, epochs, learning_rate, batch_size=batch_size)
    except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError) as e:
        status = f"failed: {e!s}"

    return {"backend": "jax", "action": "dpo", "model": model_name, "dataset": dataset, "beta": beta, "status": status, "final_loss": final_loss}

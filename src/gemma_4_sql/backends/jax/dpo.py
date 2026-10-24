"""JAX-specific DPO (Direct Preference Optimization) logic."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gemma_4_sql.backends.common_dpo import generic_dpo_loss
from gemma_4_sql.backends.common_train import generic_run_training_epochs
from gemma_4_sql.backends.jax.etl import build_dataloader
from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.exceptions import DependencyMissingError, TrainingError
from gemma_4_sql.type_hints import DPOConfig, ETLConfig, ModelType, TensorType, TrainerState

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from gemma_4_sql.type_hints import DPOResultDict, JSONDict

jax = optional_import("jax")
jnn = optional_import("jax.nn")
jnp = optional_import("jax.numpy")
optax = optional_import("optax")
nnx = optional_import("flax.nnx")
Gemma4Config = optional_attr("gemma_4_sql.backends.jax.gemma4", "Gemma4Config")
Gemma4ForCausalLM = optional_attr("gemma_4_sql.backends.jax.gemma4", "Gemma4ForCausalLM")


def _create_nnx_optimizer(model: Any, tx: Any) -> Any:
    """Create a Flax NNX Optimizer compatible with both Flax <0.11 and >=0.11 (`wrt=nnx.Param`).

    Args:
        model: Flax NNX module to optimize.
        tx: Optax gradient transformation.

    Returns:
        Configured `nnx.Optimizer` instance.

    """
    try:
        return nnx.Optimizer(model, tx, wrt=nnx.Param)
    except TypeError:
        return nnx.Optimizer(model, tx)


def _apply_nnx_optimizer_update(optimizer: Any, model: Any, grads: Any) -> None:
    """Apply gradient updates on an `nnx.Optimizer` across Flax NNX versions.

    Args:
        optimizer: Flax NNX optimizer instance.
        model: Flax NNX module being optimized.
        grads: Computed gradient PyTree.

    """
    try:
        optimizer.update(model, grads)
    except TypeError:
        optimizer.update(grads)


def dpo_loss(
    policy_chosen_logps: TensorType,
    policy_rejected_logps: TensorType,
    ref_chosen_logps: TensorType,
    ref_rejected_logps: TensorType,
    beta: float = 0.1,
) -> tuple[Any, Any, Any]:
    """Compute the Direct Preference Optimization (DPO) loss for JAX arrays.

    Args:
        policy_chosen_logps: Log probabilities of the chosen completions from the policy model.
        policy_rejected_logps: Log probabilities of the rejected completions from the policy model.
        ref_chosen_logps: Log probabilities of the chosen completions from the reference model.
        ref_rejected_logps: Log probabilities of the rejected completions from the reference model.
        beta: The beta parameter controlling the KL penalty.

    Returns:
        A 3-tuple of `(mean_loss, chosen_rewards, rejected_rewards)`.

    """
    if jnp is None or jnn is None:
        return (0.0, 0.0, 0.0)
    return generic_dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta, jnn.log_sigmoid)


def _compute_logps(model: ModelType, inputs: Any, labels: Any) -> Any:
    """Compute sequence log probabilities for DPO using token-level log-softmax.

    Args:
        model: Policy or reference model callable.
        inputs: Input token ID tensor of shape `(batch_size, seq_len)`.
        labels: Target token ID tensor of shape `(batch_size, seq_len)`.

    Returns:
        Summed non-padding token log probabilities per sequence.

    """
    try:
        seq_len = inputs.shape[1] if hasattr(inputs, "shape") and len(inputs.shape) > 1 else 1
        positions = jnp.arange(seq_len)[None, :] if hasattr(jnp, "arange") else None
        logits = model(inputs, positions) if positions is not None else model(inputs)
    except TypeError:
        logits = model(inputs)

    log_probs = jnn.log_softmax(logits, axis=-1)
    labels_expanded = jnp.expand_dims(labels, axis=-1)
    selected_log_probs = jnp.take_along_axis(log_probs, labels_expanded, axis=-1)
    selected_log_probs = jnp.squeeze(selected_log_probs, axis=-1)
    mask = labels != 0
    return jnp.sum(selected_log_probs * mask, axis=-1)


def _dpo_step_loss(policy_model: ModelType, ref_model: ModelType, batch: Mapping[str, Any], beta: float) -> Any:
    """Compute scalar DPO loss for a preference batch.

    Args:
        policy_model: Trainable policy model.
        ref_model: Frozen reference model.
        batch: Preference batch dictionary with chosen/rejected inputs and labels.
        beta: KL penalty temperature coefficient.

    Returns:
        Scalar DPO loss value.

    """
    ch_inputs = batch.get("chosen_inputs", batch.get("chosen_input_ids"))
    re_inputs = batch.get("rejected_inputs", batch.get("rejected_input_ids"))
    ch_labels = batch["chosen_labels"]
    re_labels = batch["rejected_labels"]
    pi_ch_logps = _compute_logps(policy_model, ch_inputs, ch_labels)
    pi_re_logps = _compute_logps(policy_model, re_inputs, re_labels)
    ref_ch_logps = _compute_logps(ref_model, ch_inputs, ch_labels)
    ref_re_logps = _compute_logps(ref_model, re_inputs, re_labels)
    (loss, _, _) = dpo_loss(pi_ch_logps, pi_re_logps, ref_ch_logps, ref_re_logps, beta)
    return loss


def _get_train_step_fn(beta: float) -> Callable[..., Any]:
    """Return a JIT-compiled DPO training step function for the given beta.

    Args:
        beta: KL penalty temperature coefficient.

    Returns:
        Compiled or eager training step callable.

    Raises:
        DependencyMissingError: If `flax.nnx.value_and_grad` is unavailable.

    """
    if nnx is None or not hasattr(nnx, "value_and_grad"):
        raise DependencyMissingError(
            "Flax NNX value_and_grad is required for JAX DPO.",
            dependency_name="flax.nnx",
            backend="jax",
        )

    def train_step(policy_model: ModelType, ref_model: ModelType, optimizer: Any, batch: JSONDict) -> Any:
        """Execute a single JAX-compiled DPO training step.

        Args:
            policy_model: Trainable policy model.
            ref_model: Frozen reference model.
            optimizer: Flax NNX optimizer instance.
            batch: Preference batch dictionary.

        Returns:
            Scalar step loss.

        """

        def _loss_wrapper(p: Any, r: Any, b: Any) -> Any:
            """Evaluate _dpo_step_loss for gradient transformation."""
            return _dpo_step_loss(p, r, b, beta)

        (loss, grads) = nnx.value_and_grad(_loss_wrapper)(policy_model, ref_model, batch)
        _apply_nnx_optimizer_update(optimizer, policy_model, grads)
        return loss

    if hasattr(nnx, "jit"):
        return nnx.jit(train_step)
    return train_step


def _run_training_epochs(state: TrainerState[Any, Any, Any, Any]) -> float:
    """Run DPO training epochs over the preference dataloader.

    Args:
        state: `TrainerState` holding models, optimizer, dataloader, and step function.

    Returns:
        Mean batch loss from the final epoch.

    """
    dataloader: Iterable[JSONDict] = state.dataloader or ()
    policy_model = state.policy_model
    ref_model = state.ref_model
    optimizer = state.optimizer
    train_step = state.train_step

    def _step(batch: JSONDict) -> float:
        """Execute a single JAX DPO step and extract scalar float loss."""
        assert train_step is not None
        loss = train_step(policy_model, ref_model, optimizer, batch)
        return float(loss.item() if hasattr(loss, "item") else loss)

    return generic_run_training_epochs(state.epochs, dataloader, _step)


def _resolve_gemma4_config(model_name: str) -> Any:
    """Resolve a `Gemma4Config` preset for the given `model_name`.

    Args:
        model_name: Model name or preset identifier.

    Returns:
        Resolved `Gemma4Config` object.

    """
    return Gemma4Config.from_preset(model_name) if hasattr(Gemma4Config, "from_preset") else Gemma4Config.gemma4_e2b()


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
        raise DependencyMissingError("JAX dependencies are missing for DPO.")

    resolved_cfg = _resolve_gemma4_config(model_name)
    policy_model = Gemma4ForCausalLM(resolved_cfg, rngs=nnx.Rngs(0))
    ref_model = Gemma4ForCausalLM(resolved_cfg, rngs=nnx.Rngs(1))
    optimizer = _create_nnx_optimizer(policy_model, optax.adamw(learning_rate))
    train_step = _get_train_step_fn(beta)
    data_dict = build_dataloader(ETLConfig(dataset_name=dataset, split="train", batch_size=batch_size))
    dataloader = data_dict.get("loader", None)

    if dataloader is None or not hasattr(dataloader, "__iter__"):
        raise ValueError(f"Invalid dataloader for dataset: {dataset}")

    final_loss = _run_training_epochs(
        TrainerState(
            dataloader=dataloader,
            epochs=epochs,
            policy_model=policy_model,
            ref_model=ref_model,
            optimizer=optimizer,
            train_step=train_step,
        )
    )
    return "completed", final_loss


def run_dpo(config: DPOConfig, **kwargs: object) -> DPOResultDict:
    """Run a DPO training loop for JAX.

    Args:
        config: DPO training configuration.
        **kwargs: Hyperparameters for DPO (e.g., beta, learning_rate, raise_on_error).

    Returns:
        Structured `DPOResultDict` with execution status and metrics.

    Raises:
        DependencyMissingError: If JAX dependencies are missing.
        TrainingError: If DPO fails and `raise_on_error` is enabled.

    """
    model_name = getattr(config, "model_name", "model")
    dataset = getattr(config, "dataset", "dataset")
    beta = getattr(config, "beta", 0.1)
    epochs = getattr(config, "epochs", 1)
    learning_rate = getattr(config, "learning_rate", 1e-05)
    batch_size = getattr(config, "batch_size", 2)
    extra = getattr(config, "extra_kwargs", {}) or {}
    raise_on_error = bool(kwargs.get("raise_on_error", extra.get("raise_on_error", False)))
    final_loss = 0.0
    status = "completed"
    if jax is None or jnp is None or jnn is None or optax is None or Gemma4ForCausalLM is None:
        raise DependencyMissingError("JAX DPO dependencies are missing.", backend="jax")

    try:
        status, final_loss = _execute_dpo(model_name, dataset, beta, epochs, learning_rate, batch_size=batch_size)
    except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError) as e:
        if raise_on_error:
            raise TrainingError(
                f"JAX DPO failed: {e!s}",
                backend="jax",
                model_name=model_name,
                operation="dpo",
                cause=e,
            ) from e
        status = f"failed: {e!s}"

    return {"backend": "jax", "action": "dpo", "model": model_name, "dataset": dataset, "beta": beta, "status": status, "final_loss": final_loss}

"""MaxText-specific DPO (Direct Preference Optimization) logic."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from gemma_4_sql.backends.common_dpo import run_dpo_wrapper
from gemma_4_sql.backends.common_train import generic_run_training_epochs
from gemma_4_sql.backends.jax.dpo import dpo_loss as jax_dpo_loss
from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.backends.maxtext.etl import build_dataloader
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import DPOConfig, DPOResultDict, ETLConfig, MaxTextCausalLMProtocol, MaxTextTrainerState, TensorType, TrainerState

logger = logging.getLogger(__name__)

jax = optional_import("jax")
jnn = optional_import("jax.nn")
jnp = optional_import("jax.numpy")
optax = optional_import("optax")
Gemma4Model = optional_attr("maxtext.models.gemma4", "Gemma4Model")


def dpo_loss(
    policy_chosen_logps: TensorType,
    policy_rejected_logps: TensorType,
    ref_chosen_logps: TensorType,
    ref_rejected_logps: TensorType,
    beta: float = 0.1,
) -> tuple[object, object, object]:
    """Compute the DPO loss for MaxText using JAX under the hood.

    Args:
        policy_chosen_logps: Log probabilities of the chosen completions from the policy model.
        policy_rejected_logps: Log probabilities of the rejected completions from the policy model.
        ref_chosen_logps: Log probabilities of the chosen completions from the reference model.
        ref_rejected_logps: Log probabilities of the rejected completions from the reference model.
        beta: The beta parameter controlling the KL penalty.

    Returns:
        A 3-tuple of `(mean_loss, chosen_rewards, rejected_rewards)`.

    """
    return jax_dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta)


def _compute_logps(
    model: MaxTextCausalLMProtocol,
    params: dict[str, TensorType],
    inputs: TensorType,
    labels: TensorType,
) -> TensorType:
    """Compute sequence log probabilities from logits and target labels.

    Applies `jax.nn.log_softmax` over the vocabulary dimension, gathers log-probabilities
    at the target token indices via `jnp.take_along_axis`, masks out padding tokens
    (`labels != 0`), and sums along the sequence axis.

    Args:
        model: Stateless Flax/MaxText neural network model instance.
        params: Model parameters dictionary.
        inputs: Input token IDs tensor of shape `(batch_size, seq_len)`.
        labels: Target label IDs tensor of shape `(batch_size, seq_len)`.

    Returns:
        Summed non-padding log probability per sequence.

    """
    logits: Any = model.apply(params, inputs)
    ndim = getattr(logits, "ndim", 0)
    if jnn is not None and jnp is not None and isinstance(ndim, int) and ndim >= 2 and hasattr(jnn, "log_softmax") and hasattr(jnp, "take_along_axis"):
        log_probs = jnn.log_softmax(logits, axis=-1)
        labels_expanded = jnp.expand_dims(labels, axis=-1)
        selected_log_probs = jnp.squeeze(jnp.take_along_axis(log_probs, labels_expanded, axis=-1), axis=-1)
        labels_any: Any = labels
        mask = labels_any != 0
        summed: TensorType = jnp.sum(selected_log_probs * mask, axis=-1)
        return summed
    fallback: TensorType = jnp.sum(logits * labels, axis=-1) if jnp is not None else logits
    return fallback


def _dpo_step_loss(
    policy_model: MaxTextCausalLMProtocol,
    policy_params: dict[str, TensorType],
    ref_model: MaxTextCausalLMProtocol,
    ref_params: dict[str, TensorType],
    batch: Mapping[str, TensorType],
    beta: float,
) -> object:
    """Calculate single-step DPO loss.

    Args:
        policy_model: Policy model instance.
        policy_params: Policy parameters.
        ref_model: Reference model instance.
        ref_params: Reference parameters.
        batch: Mapping containing chosen and rejected inputs/labels tensors.
        beta: KL temperature penalty factor.

    Returns:
        Calculated step loss tensor.

    """
    pi_ch_logps = _compute_logps(policy_model, policy_params, batch["chosen_inputs"], batch["chosen_labels"])
    pi_re_logps = _compute_logps(policy_model, policy_params, batch["rejected_inputs"], batch["rejected_labels"])
    ref_ch_logps = _compute_logps(ref_model, ref_params, batch["chosen_inputs"], batch["chosen_labels"])
    ref_re_logps = _compute_logps(ref_model, ref_params, batch["rejected_inputs"], batch["rejected_labels"])
    (loss, _, _) = dpo_loss(pi_ch_logps, pi_re_logps, ref_ch_logps, ref_re_logps, beta)
    return loss


def _get_train_step_fn(
    policy_model: MaxTextCausalLMProtocol,
    ref_model: MaxTextCausalLMProtocol,
    optimizer: object,
    beta: float,
) -> Callable[..., tuple[Any, Any, Any]]:
    """Create a JIT-compiled train step function.

    Args:
        policy_model: Policy model instance.
        ref_model: Reference model instance.
        optimizer: Optax optimizer instance.
        beta: Beta KL penalty parameter.

    Returns:
        JIT-compiled training step function.

    """

    def train_step(
        policy_params: dict[str, TensorType],
        ref_params: dict[str, TensorType],
        opt_state: dict[str, TensorType],
        batch: Mapping[str, TensorType],
    ) -> tuple[Any, Any, Any]:
        """Execute one optimization step.

        Args:
            policy_params: Policy parameters.
            ref_params: Reference parameters.
            opt_state: Optimizer state.
            batch: Data batch mapping.

        Returns:
            Tuple of updated policy parameters, optimizer state, and loss.

        """

        def _wrapper(
            p: dict[str, TensorType],
            r: dict[str, TensorType],
            b: Mapping[str, TensorType],
        ) -> object:
            """Compute DPO step loss for gradient differentiation."""
            return _dpo_step_loss(policy_model, p, ref_model, r, b, beta)

        opt_obj: Any = optimizer
        (loss, grads) = jax.value_and_grad(_wrapper)(policy_params, ref_params, batch)
        (updates, new_opt_state) = opt_obj.update(grads, opt_state, policy_params)
        updated_params = optax.apply_updates(policy_params, updates)
        return (updated_params, new_opt_state, loss)

    if jax is not None and hasattr(jax, "jit"):
        jit_fn: Callable[..., tuple[Any, Any, Any]] = jax.jit(train_step)
        return jit_fn
    return train_step


def _run_training_epochs(state: MaxTextTrainerState[Any, Any, Any] | TrainerState[Any, Any, Any, Any]) -> tuple[Any, Any, float]:
    """Run DPO training epochs for MaxText.

    Args:
        state: MaxTextTrainerState containing dataloader and parameters.

    Returns:
        A 3-tuple of `(policy_params, opt_state, final_loss)`.

    """
    policy_params = state.policy_params
    opt_state = state.opt_state
    step_fn = state.train_step
    dataloader: Iterable[dict[str, TensorType]] = state.dataloader or ()

    def process_batch(batch: dict[str, TensorType]) -> float:
        """Process a single training batch.

        Args:
            batch: Dictionary containing training batch tensors.

        Returns:
            Computed batch loss scalar.

        """
        nonlocal policy_params, opt_state
        assert step_fn is not None
        (policy_params, opt_state, loss) = step_fn(policy_params, state.ref_params, opt_state, batch)
        return float(loss.item() if hasattr(loss, "item") else loss)

    final_loss = generic_run_training_epochs(state.epochs, dataloader, process_batch)
    return (policy_params, opt_state, final_loss)


def _execute_dpo(
    model_name: str,
    dataset: str,
    beta: float,
    epochs: int,
    learning_rate: float,
    batch_size: int = 2,
) -> tuple[str, float]:
    """Execute the core DPO loop.

    Args:
        model_name: Target model identifier.
        dataset: Dataset identifier.
        beta: KL penalty weight.
        epochs: Number of training epochs.
        learning_rate: Optimizer learning rate.
        batch_size: Training batch size.

    Returns:
        Tuple of completion status and final loss.

    Raises:
        ValueError: If dataloader could not be constructed.

    """
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
    data_dict = build_dataloader(ETLConfig(dataset_name=dataset, split="train", batch_size=batch_size))
    dataloader: Any = data_dict.get("loader", None)

    if dataloader is None or not hasattr(dataloader, "__iter__"):
        raise ValueError(f"Invalid dataloader for dataset: {dataset}")

    (policy_params, opt_state, final_loss) = _run_training_epochs(
        MaxTextTrainerState(
            dataloader=dataloader,
            epochs=epochs,
            train_step=train_step,
            policy_params=policy_params,
            ref_params=ref_params,
            opt_state=opt_state,
        )
    )
    return "completed", final_loss


def run_dpo(config: DPOConfig, **kwargs: object) -> DPOResultDict:
    """Run DPO training loop for MaxText.

    Args:
        config: DPO configuration object.
        **kwargs: Extra parameters (such as `raise_on_error=True`).

    Returns:
        Structured `DPOResultDict` with the execution status and metrics.

    Raises:
        DependencyMissingError: If MaxText or JAX dependencies are missing.
        TrainingError: If DPO fails and `raise_on_error` is enabled.

    """
    if jax is None or jnp is None or optax is None or Gemma4Model is None:
        raise DependencyMissingError("MaxText dependencies are missing.", backend="maxtext")
    return run_dpo_wrapper(
        "maxtext",
        config,
        lambda m, d, b, ep, lr, bs: _execute_dpo(m, d, b, ep, lr, batch_size=bs),
        **kwargs,
    )

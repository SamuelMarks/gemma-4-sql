"""MaxText-specific DPO (Direct Preference Optimization) logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from gemma_4_sql.backends.common_train import generic_run_training_epochs
from gemma_4_sql.backends.jax.dpo import dpo_loss as jax_dpo_loss
from gemma_4_sql.backends.maxtext.etl import build_dataloader
from gemma_4_sql.type_hints import DPOConfig, ETLConfig, JSONDict, TrainerState

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

try:
    import jax as _jax
    import jax.numpy as _jnp
    import optax as _optax
    from maxtext.models.gemma4 import Gemma4Model as _Gemma4Model

    jax: Any = _jax
    jnp: Any = _jnp
    optax: Any = _optax
    Gemma4Model: Any = _Gemma4Model
except (ImportError, AttributeError):
    jax = None
    jnp = None
    optax = None
    Gemma4Model = None


def dpo_loss(
    policy_chosen_logps: object,
    policy_rejected_logps: object,
    ref_chosen_logps: object,
    ref_rejected_logps: object,
    beta: float = 0.1,
) -> tuple[object, object, object]:
    """Compute the DPO loss for MaxText (using JAX under the hood).

    Args:
        policy_chosen_logps: Log probabilities of the chosen completions from the policy model.
        policy_rejected_logps: Log probabilities of the rejected completions from the policy model.
        ref_chosen_logps: Log probabilities of the chosen completions from the reference model.
        ref_rejected_logps: Log probabilities of the rejected completions from the reference model.
        beta: The beta parameter controlling the KL penalty.

    Returns:
        A tuple containing the results.
    """
    return jax_dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta)


def _compute_logps(model: Any, params: Any, inputs: Any, labels: Any) -> Any:
    """Compute sequence log probabilities from logits and labels.

    Args:
        model: Neural network model instance.
        params: Model parameters.
        inputs: Input token ids.
        labels: Target label ids.

    Returns:
        Log probability sum per sequence.
    """
    logits = model.apply(params, inputs)
    return jnp.sum(logits * labels, axis=-1)


def _dpo_step_loss(
    policy_model: Any,
    policy_params: Any,
    ref_model: Any,
    ref_params: Any,
    batch: JSONDict,
    beta: float,
) -> Any:
    """Calculate single-step DPO loss.

    Args:
        policy_model: Policy model instance.
        policy_params: Policy parameters.
        ref_model: Reference model instance.
        ref_params: Reference parameters.
        batch: Dictionary containing chosen and rejected inputs/labels.
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


def _get_train_step_fn(policy_model: Any, ref_model: Any, optimizer: Any, beta: float) -> Callable[..., Any]:
    """Create a JIT-compiled train step function.

    Args:
        policy_model: Policy model instance.
        ref_model: Reference model instance.
        optimizer: Optax optimizer instance.
        beta: Beta KL penalty parameter.

    Returns:
        JIT-compiled training step function.
    """

    def train_step(policy_params: Any, ref_params: Any, opt_state: Any, batch: JSONDict) -> tuple[Any, Any, Any]:
        """Execute one optimization step.

        Args:
            policy_params: Policy parameters.
            ref_params: Reference parameters.
            opt_state: Optimizer state.
            batch: Data batch.

        Returns:
            Tuple of updated policy parameters, optimizer state, and loss.
        """
        (loss, grads) = jax.value_and_grad(lambda p, r, b: _dpo_step_loss(policy_model, p, ref_model, r, b, beta))(policy_params, ref_params, batch)
        (updates, opt_state) = optimizer.update(grads, opt_state, policy_params)
        policy_params = optax.apply_updates(policy_params, updates)
        return (policy_params, opt_state, loss)

    if jax is not None and hasattr(jax, "jit"):
        return jax.jit(train_step)
    return train_step


def _run_training_epochs(state: TrainerState) -> tuple[Any, Any, float]:
    """Run training epochs.

    Args:
        state: Trainer state containing dataloader and parameters.

    Returns:
        tuple: (policy_params, opt_state, final_loss)
    """
    policy_params = state.policy_params
    opt_state = state.opt_state

    def process_batch(batch: Any) -> float:
        """Process a single training batch.

        Args:
            batch: Dictionary containing training batch.

        Returns:
            Computed batch loss.
        """
        nonlocal policy_params, opt_state
        (policy_params, opt_state, loss) = state.train_step(policy_params, state.ref_params, opt_state, batch)
        return float(loss.item() if hasattr(loss, "item") else loss)

    final_loss = generic_run_training_epochs(state.epochs, state.dataloader, process_batch)
    return (policy_params, opt_state, final_loss)


def _execute_dpo(
    model_name: str,
    dataset: str,
    beta: float,
    epochs: int,
    learning_rate: float,
    test_mode: bool,
    batch_size: int = 2,
) -> tuple[str, float]:
    """Execute the core DPO loop.

    Args:
        model_name: Target model identifier.
        dataset: Dataset identifier.
        beta: KL penalty weight.
        epochs: Number of training epochs.
        learning_rate: Optimizer learning rate.
        test_mode: Boolean indicating testing mode.
        batch_size: Training batch size.

    Returns:
        Tuple of completion status and final loss.

    Raises:
        ValueError: If dataloader could not be constructed.
    """
    if not test_mode:  # pragma: no cover
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
    dataloader = data_dict.get("loader", None)

    if dataloader is None or not hasattr(dataloader, "__iter__"):
        raise ValueError(f"Invalid dataloader for dataset: {dataset}")

    (policy_params, opt_state, final_loss) = _run_training_epochs(
        TrainerState(
            dataloader=dataloader,
            epochs=epochs,
            train_step=train_step,
            policy_params=policy_params,
            ref_params=ref_params,
            opt_state=opt_state,
        )
    )
    return "completed", final_loss


def run_dpo(config: DPOConfig, **kwargs: object) -> JSONDict:
    """Run DPO training loop for MaxText.

    Args:
        config: DPO configuration object.
        **kwargs: Extra parameters like test_mode.

    Returns:
        A dict with the execution status and metrics.

    Raises:
        DependencyMissingError: If MaxText or JAX dependencies are missing.
    """
    model_name = getattr(config, "model_name", "model")
    dataset = getattr(config, "dataset", "dataset")
    beta = getattr(config, "beta", 0.1)
    epochs = getattr(config, "epochs", 1)
    learning_rate = getattr(config, "learning_rate", 1e-05)

    final_loss = 0.0
    status = "completed"
    if jax is None or jnp is None or optax is None or Gemma4Model is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("MaxText dependencies are missing.")
    try:
        batch_size = getattr(config, "batch_size", 2)
        status, final_loss = _execute_dpo(
            model_name,
            dataset,
            beta,
            epochs,
            learning_rate,
            bool(kwargs.get("test_mode")),
            batch_size=batch_size,
        )
    except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError) as e:
        logger.exception("DPO Train error: ")
        status = f"failed: {e!s}"
    return {
        "backend": "maxtext",
        "action": "dpo",
        "model": model_name,
        "dataset": dataset,
        "beta": beta,
        "status": status,
        "final_loss": final_loss,
    }

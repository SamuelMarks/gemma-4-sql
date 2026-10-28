"""MaxText-specific DPO (Direct Preference Optimization) logic."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Iterable, Mapping
from typing import Protocol, SupportsFloat, runtime_checkable

from gemma_4_sql.backends.common_dpo import (
    ArithmeticTensorProtocol,
    gather_jax_sequence_logps,
    run_dpo_wrapper,
)
from gemma_4_sql.backends.common_train import generic_run_training_epochs
from gemma_4_sql.backends.jax.dpo import dpo_loss as jax_dpo_loss
from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.backends.maxtext.etl import build_dataloader
from gemma_4_sql.exceptions import DependencyMissingError, StatusValue
from gemma_4_sql.type_hints import (
    DPOConfig,
    DPOResultDict,
    ETLConfig,
    JaxArrayProtocol,
    MaxTextCausalLMProtocol,
    MaxTextTrainerState,
    PyTreeNode,
    TensorProtocol,
    TensorType,
    TrainerState,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class _StatelessApplyModelProtocol[ValT](Protocol):
    """Protocol for a stateless Flax/MaxText model exposing `.apply(params, inputs)`."""

    def apply(
        self,
        params: Mapping[str, ValT],
        inputs: ValT,
        /,
    ) -> ValT:
        """Apply model parameters to `inputs`."""
        ...


type MaxTextDPOModel[ValT] = _StatelessApplyModelProtocol[ValT] | MaxTextCausalLMProtocol | str


@runtime_checkable
class _OptaxOptimizerProtocol[OptInputT, OptStateT](Protocol):
    """Protocol for an Optax `GradientTransformation` supporting `.init(...)` and `.update(...)`."""

    def init(self, params: OptInputT, /) -> OptStateT:
        """Initialize optimizer state from parameters."""
        ...

    def update(
        self,
        updates: OptInputT,
        state: OptStateT,
        params: OptInputT | None = None,
        /,
    ) -> tuple[OptInputT, OptStateT]:
        """Compute parameter updates and next optimizer state."""
        ...


type MaxTextDPOStepCallable[ParamsT, OptStateT, BatchT] = Callable[
    [
        ParamsT,
        ParamsT,
        OptStateT,
        BatchT,
    ],
    tuple[ParamsT, OptStateT, SupportsFloat | float],
]


jax = optional_import("jax")
jnn = optional_import("jax.nn")
jnp = optional_import("jax.numpy")
optax = optional_import("optax")
_raw_gemma4 = optional_attr("maxtext.models.gemma4", "Gemma4Model")
Gemma4Model: Callable[..., MaxTextCausalLMProtocol] | None = _raw_gemma4 if callable(_raw_gemma4) else None  # ty: ignore[invalid-assignment]


def dpo_loss[TensorT: ArithmeticTensorProtocol | float](
    policy_chosen_logps: TensorT,
    policy_rejected_logps: TensorT,
    ref_chosen_logps: TensorT,
    ref_rejected_logps: TensorT,
    beta: float = 0.1,
) -> tuple[TensorT | ArithmeticTensorProtocol | float, TensorT, TensorT]:
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


def _compute_logps[ValT: TensorType | SupportsFloat](
    model: MaxTextDPOModel[ValT],
    params: Mapping[str, ValT],
    inputs: ValT,
    labels: ValT,
) -> ValT:
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
    logits = model.apply(params, inputs)  # pyright: ignore[reportAttributeAccessIssue,reportArgumentType]  # ty: ignore[unresolved-attribute, invalid-argument-type]
    if jnn is not None and jnp is not None and isinstance(logits, JaxArrayProtocol) and logits.ndim >= 2:
        with contextlib.suppress(AttributeError, TypeError):
            summed: ValT = gather_jax_sequence_logps(jnn, jnp, logits, labels)  # pyright: ignore[reportArgumentType,reportAssignmentType]  # ty: ignore[invalid-assignment, invalid-argument-type]
            return summed
    fallback: ValT = jnp.sum(logits * labels, axis=-1) if jnp is not None else logits  # pyright: ignore[reportOperatorIssue,reportAssignmentType]  # ty: ignore[invalid-assignment, unsupported-operator]
    return fallback


def _dpo_step_loss[ValT: ArithmeticTensorProtocol](
    policy_model: MaxTextDPOModel[ValT],
    policy_params: Mapping[str, ValT],
    ref_model: MaxTextDPOModel[ValT],
    ref_params: Mapping[str, ValT],
    batch: Mapping[str, ValT],
    beta: float,
) -> ValT | ArithmeticTensorProtocol | float:
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


def _get_train_step_fn[ValT: ArithmeticTensorProtocol, ParamsT, OptInputT, OptStateT, BatchT](
    policy_model: MaxTextDPOModel[ValT],
    ref_model: MaxTextDPOModel[ValT],
    optimizer: _OptaxOptimizerProtocol[OptInputT, OptStateT],
    beta: float,
) -> MaxTextDPOStepCallable[ParamsT, OptStateT, BatchT]:
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
        policy_params: ParamsT,
        ref_params: ParamsT,
        opt_state: OptStateT,
        batch: BatchT,
    ) -> tuple[
        ParamsT,
        OptStateT,
        SupportsFloat | float,
    ]:
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
            p: ParamsT,
            r: ParamsT,
            b: BatchT,
        ) -> TensorType | ArithmeticTensorProtocol | float:
            """Compute DPO step loss for gradient differentiation."""
            return _dpo_step_loss(policy_model, p, ref_model, r, b, beta)  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]

        assert jax is not None
        assert optax is not None
        (loss, grads) = jax.value_and_grad(_wrapper)(policy_params, ref_params, batch)
        (updates, new_opt_state) = optimizer.update(grads, opt_state, policy_params)  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]
        updated_params: ParamsT = optax.apply_updates(policy_params, updates)  # pyright: ignore[reportAssignmentType]
        return (updated_params, new_opt_state, loss)

    with contextlib.suppress(AttributeError, TypeError):
        jit_fn: MaxTextDPOStepCallable[ParamsT, OptStateT, BatchT] = jax.jit(train_step)  # pyright: ignore[reportOptionalMemberAccess]  # ty: ignore[unresolved-attribute]
        return jit_fn
    return train_step


def _run_training_epochs[ModelT, ParamsT, OptT, BatchT](
    state: MaxTextTrainerState[ParamsT, OptT, BatchT] | TrainerState[ModelT, ParamsT, OptT, BatchT],
) -> tuple[ParamsT | None, OptT | None, float]:
    """Run DPO training epochs for MaxText.

    Args:
        state: `MaxTextTrainerState` containing dataloader and parameters.

    Returns:
        A 3-tuple of `(policy_params, opt_state, final_loss)`.

    """
    policy_params = state.policy_params
    opt_state = state.opt_state
    step_fn = state.train_step
    dataloader: Iterable[BatchT] = state.dataloader or ()

    def process_batch(batch: BatchT) -> float:
        """Process a single training batch.

        Args:
            batch: Dictionary containing training batch tensors.

        Returns:
            Computed batch loss scalar.

        """
        nonlocal policy_params, opt_state
        assert step_fn is not None
        step_out: tuple[ParamsT | None, OptT | None, SupportsFloat | float] = step_fn(  # ty: ignore[invalid-assignment]
            policy_params,
            state.ref_params,
            opt_state,
            batch,
        )  # pyright: ignore[reportAssignmentType]
        policy_params, opt_state, loss = step_out
        try:
            return float(loss.item())  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]
        except (AttributeError, TypeError, ValueError):
            return float(loss)

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
    assert jax is not None
    assert jnp is not None
    assert optax is not None
    assert Gemma4Model is not None
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
    optimizer: _OptaxOptimizerProtocol[
        Mapping[str, TensorProtocol | Mapping[str, TensorProtocol]],
        Mapping[str, TensorProtocol | Mapping[str, TensorProtocol]],
    ] = optax.adamw(learning_rate)
    opt_state = optimizer.init(policy_params)
    train_step = _get_train_step_fn(policy_model, ref_model, optimizer, beta)
    data_dict = build_dataloader(ETLConfig(dataset_name=dataset, split="train", batch_size=batch_size))
    dataloader = data_dict.get("loader", None)
    has_loader = isinstance(dataloader, Iterable)

    if not has_loader:
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


def run_dpo(config: DPOConfig, **kwargs: StatusValue) -> DPOResultDict:
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


__all__ = [
    "Gemma4Model",
    "PyTreeNode",
    "dpo_loss",
    "run_dpo",
]

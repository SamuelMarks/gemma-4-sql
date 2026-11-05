"""MaxText-specific DPO (Direct Preference Optimization) logic."""

from __future__ import annotations

import contextlib
import logging
import types
from collections.abc import Callable, Iterable, Mapping
from typing import Protocol, SupportsFloat, TypeGuard, runtime_checkable

from gemma_4_sql._types_models import MaxTextModelConstructorProtocol, StepResult
from gemma_4_sql.backends.common_dpo import (
    ArithmeticTensorProtocol,
    JaxNNLogSoftmaxProtocol,
    JaxNumpyLogpOpsProtocol,
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
    ScalarItemExtractorProtocol,
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


type MaxTextDPOTarget = Mapping[str, PyTreeNode] | Mapping[str, JaxArrayProtocol] | Mapping[str, TensorType] | Mapping[str, ArithmeticTensorProtocol] | PyTreeNode | JaxArrayProtocol | TensorType | ArithmeticTensorProtocol | SupportsFloat

type MaxTextDPOStepCallable[ParamsT, OptStateT, BatchT] = Callable[
    [
        ParamsT,
        ParamsT,
        OptStateT,
        BatchT,
    ],
    tuple[ParamsT, OptStateT, SupportsFloat | float],
]


def _is_stateless_model[ValT](
    model: MaxTextDPOModel[ValT],
    _sample: ValT,
) -> TypeGuard[_StatelessApplyModelProtocol[ValT]]:
    """Narrow `model` to `_StatelessApplyModelProtocol[ValT]`.

    Args:
        model: Target stateless model instance.
        _sample: Sample value for type parameter binding.

    Returns:
        `True` if `model` is not a plain `str`.

    """
    _ = _sample
    return not isinstance(model, str)


def _is_jnn_gather(
    mod: types.ModuleType | None,
) -> TypeGuard[JaxNNLogSoftmaxProtocol[JaxArrayProtocol]]:
    """Narrow `mod` to `JaxNNLogSoftmaxProtocol[JaxArrayProtocol]`.

    Args:
        mod: Loaded `jax.nn` module.

    Returns:
        `True` when `mod` is loaded.

    """
    return mod is not None


def _is_jnp_gather(
    mod: types.ModuleType | None,
) -> TypeGuard[JaxNumpyLogpOpsProtocol[JaxArrayProtocol]]:
    """Narrow `mod` to `JaxNumpyLogpOpsProtocol[JaxArrayProtocol]`.

    Args:
        mod: Loaded `jax.numpy` module.

    Returns:
        `True` when `mod` is loaded.

    """
    return mod is not None


def _is_arithmetic_val(
    val: TensorType | SupportsFloat,
) -> TypeGuard[JaxArrayProtocol]:
    """Narrow `val` to `JaxArrayProtocol` for elementwise multiplication.

    Args:
        val: Candidate tensor or numeric scalar.

    Returns:
        `True` if `val` is not a plain `str`.

    """
    return not isinstance(val, str)


def _as_val_t[ValT](
    val: MaxTextDPOTarget,
    _sample: ValT,
) -> TypeGuard[ValT]:
    """Narrow `val` to `ValT`.

    Args:
        val: Candidate result tensor.
        _sample: Sample value of type `ValT`.

    Returns:
        `True` if `val` is not a plain `str`.

    """
    _ = _sample
    return not isinstance(val, str)


def _is_val_mapping[ValT](
    obj: MaxTextDPOTarget,
    _model: MaxTextDPOModel[ValT],
) -> TypeGuard[Mapping[str, ValT]]:
    """Narrow `obj` to `Mapping[str, ValT]`.

    Args:
        obj: Parameter dictionary or batch mapping.
        _model: Model carrying type parameter `ValT`.

    Returns:
        `True` if `obj` is a `Mapping`.

    """
    _ = _model
    return isinstance(obj, Mapping)


def _is_opt_input[OptInputT, OptStateT](
    obj: MaxTextDPOTarget | None,
    _opt: _OptaxOptimizerProtocol[OptInputT, OptStateT],
) -> TypeGuard[OptInputT]:
    """Narrow `obj` to `OptInputT` for `optimizer.update(...)`.

    Args:
        obj: Gradient or parameter PyTree.
        _opt: Optimizer carrying type parameter `OptInputT`.

    Returns:
        `True` if `obj` is not `None`.

    """
    _ = _opt
    return obj is not None


def _is_params_t[ParamsT](
    obj: MaxTextDPOTarget | None,
    _sample: ParamsT,
) -> TypeGuard[ParamsT]:
    """Narrow `obj` to `ParamsT` after `optax.apply_updates`.

    Args:
        obj: Updated parameter PyTree.
        _sample: Original parameter PyTree of type `ParamsT`.

    Returns:
        `True` if `obj` is not `None`.

    """
    _ = _sample
    return obj is not None


def _is_dpo_step_tuple[ParamsT, OptT](
    res: StepResult | tuple[ParamsT, OptT, StepResult] | tuple[ParamsT | None, OptT | None, SupportsFloat | float],
    _p: ParamsT | None,
    _o: OptT | None,
) -> TypeGuard[tuple[ParamsT | None, OptT | None, ScalarItemExtractorProtocol | SupportsFloat | float]]:
    """Narrow `step_fn` output to `(policy_params, opt_state, loss)`.

    Args:
        res: Raw output from `step_fn`.
        _p: Policy parameters reference for type inference.
        _o: Optimizer state reference for type inference.

    Returns:
        `True` for 3-tuple step results.

    """
    _ = (_p, _o)
    return isinstance(res, tuple) and len(res) == 3


def _is_item_extractor(
    val: ScalarItemExtractorProtocol | SupportsFloat | float,
) -> TypeGuard[ScalarItemExtractorProtocol]:
    """Narrow `val` to `ScalarItemExtractorProtocol` for `.item()` extraction.

    Args:
        val: Loss tensor or float scalar.

    Returns:
        `True` if `val` conforms to `ScalarItemExtractorProtocol`.

    """
    return isinstance(val, ScalarItemExtractorProtocol)


def _is_supports_float(
    val: ScalarItemExtractorProtocol | SupportsFloat | float,
) -> TypeGuard[SupportsFloat | float]:
    """Narrow `val` to `SupportsFloat | float`.

    Args:
        val: Loss tensor or float scalar.

    Returns:
        `True` if `val` is a `float`, `int`, or `SupportsFloat`.

    """
    return isinstance(val, (float, int, SupportsFloat))


jax = optional_import("jax")
jnn = optional_import("jax.nn")
jnp = optional_import("jax.numpy")
optax = optional_import("optax")
Gemma4Model: MaxTextModelConstructorProtocol | None = optional_attr(
    "maxtext.models.gemma4",
    "Gemma4Model",
    MaxTextModelConstructorProtocol,
)


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


def _is_jax_mod(mod: types.ModuleType | None) -> TypeGuard[types.ModuleType]:
    """Narrow `mod` to `types.ModuleType` for attribute-guarded `.jit(...)` calls.

    Args:
        mod: Optional `jax` module.

    Returns:
        `True` when `mod` is not `None`.

    """
    return mod is not None


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
    assert _is_stateless_model(model, inputs)
    logits = model.apply(params, inputs)
    if jnn is not None and jnp is not None and isinstance(logits, JaxArrayProtocol) and logits.ndim >= 2:
        with contextlib.suppress(AttributeError, TypeError):
            assert _is_jnn_gather(jnn)
            assert _is_jnp_gather(jnp)
            lbl_arr = labels
            assert _is_arithmetic_val(lbl_arr)
            raw_gathered = gather_jax_sequence_logps(jnn, jnp, logits, lbl_arr)
            assert _as_val_t(raw_gathered, inputs)
            return raw_gathered
    assert _is_arithmetic_val(logits)
    assert _is_arithmetic_val(labels)
    raw_sum = jnp.sum(logits * labels, axis=-1) if jnp is not None else logits
    assert _as_val_t(raw_sum, inputs)
    return raw_sum


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


def _get_train_step_fn[
    ValT: ArithmeticTensorProtocol,
    ParamsT: MaxTextDPOTarget,
    OptInputT,
    OptStateT,
    BatchT: MaxTextDPOTarget,
](
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
            assert _is_val_mapping(p, policy_model)
            assert _is_val_mapping(r, ref_model)
            assert _is_val_mapping(b, policy_model)
            return _dpo_step_loss(policy_model, p, ref_model, r, b, beta)

        assert jax is not None
        assert optax is not None
        (loss, grads) = jax.value_and_grad(_wrapper)(policy_params, ref_params, batch)
        opt_policy_params = policy_params
        assert _is_opt_input(grads, optimizer)
        assert _is_opt_input(opt_policy_params, optimizer)
        (updates, new_opt_state) = optimizer.update(grads, opt_state, opt_policy_params)
        raw_updated = optax.apply_updates(opt_policy_params, updates)
        assert _is_params_t(raw_updated, policy_params)
        scalar_loss: SupportsFloat | float = loss
        return (raw_updated, new_opt_state, scalar_loss)

    with contextlib.suppress(AttributeError, TypeError, AssertionError):
        assert _is_jax_mod(jax)
        jit_fn: MaxTextDPOStepCallable[ParamsT, OptStateT, BatchT] = jax.jit(train_step)
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
        raw_step = step_fn(
            policy_params,
            state.ref_params,
            opt_state,
            batch,
        )
        assert _is_dpo_step_tuple(raw_step, policy_params, opt_state)
        policy_params, opt_state, loss = raw_step
        try:
            raw_loss = loss
            assert _is_item_extractor(raw_loss)
            return float(raw_loss.item())
        except (AttributeError, TypeError, ValueError, AssertionError):
            assert _is_supports_float(loss)
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
        dict[str, PyTreeNode],
        dict[str, PyTreeNode],
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

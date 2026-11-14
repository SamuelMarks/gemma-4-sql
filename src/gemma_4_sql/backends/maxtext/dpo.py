"""MaxText-specific DPO (Direct Preference Optimization) logic."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from typing import Protocol, SupportsFloat, runtime_checkable

from gemma_4_sql._types_models import MaxTextModelConstructorProtocol
from gemma_4_sql._types_tensors import extract_scalar_float
from gemma_4_sql.backends._jax_common import (
    JaxNNLogSoftmaxProtocol,
    JaxNumpyLogpOpsProtocol,
    JaxSliceableTensorProtocol,
    gather_jax_sequence_logps,
)
from gemma_4_sql.backends.common_dpo import (
    ArithmeticTensorProtocol,
    run_dpo_wrapper,
)
from gemma_4_sql.backends.common_train import generic_run_training_epochs
from gemma_4_sql.backends.jax.dpo import dpo_loss as jax_dpo_loss
from gemma_4_sql.backends.lazy_loader import import_optional_symbol, optional_import
from gemma_4_sql.backends.maxtext._protocols import JaxDistributedAndPRNGProtocol, JnpZerosInt32Protocol
from gemma_4_sql.backends.maxtext.etl import build_dataloader
from gemma_4_sql.exceptions import DependencyMissingError, StatusValue
from gemma_4_sql.type_hints import (
    DPOConfig,
    DPOResultDict,
    ETLConfig,
    JaxArrayProtocol,
    MaxTextDPOState,
    MlxArrayProtocol,
    PyTreeNode,
    TensorProtocol,
    TensorType,
    TfTensorProtocol,
    TorchTensorProtocol,
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
        """Apply model parameters to `inputs`.

        Args:
            params: Parameter PyTree mapping.
            inputs: Input token ID tensor.

        Returns:
            Output vocabulary logits tensor.

        """
        ...


type MaxTextDPOModel[ValT] = _StatelessApplyModelProtocol[ValT] | str


@runtime_checkable
class _OptaxOptimizerProtocol[GradInT, GradOutT, OptStateT](Protocol):
    """Protocol for an Optax `GradientTransformation` supporting `.init(...)` and `.update(...)`."""

    def init(self, params: GradInT) -> OptStateT:
        """Initialize optimizer state from parameters.

        Args:
            params: Model parameter PyTree.

        Returns:
            Initial optimizer state.

        """
        ...

    def update(
        self,
        updates: GradInT,
        state: OptStateT,
        params: GradInT | None = None,
    ) -> tuple[GradOutT, OptStateT]:
        """Compute parameter updates and next optimizer state.

        Args:
            updates: Gradient PyTree.
            state: Current optimizer state.
            params: Optional current parameter PyTree.

        Returns:
            Tuple of `(updates, next_state)`.

        """
        ...


@runtime_checkable
class _JaxJitProtocol(Protocol):
    """Protocol for a `jax` module exposing `.jit(...)`."""

    def jit[StepFnT](self, fun: StepFnT, /) -> StepFnT:
        """JIT-compile a step callable.

        Args:
            fun: Pure step function to compile.

        Returns:
            Compiled step callable.

        """
        ...


@runtime_checkable
class _JaxValueAndGrad3Protocol(Protocol):
    """Protocol for a `jax` module exposing 3-argument `.value_and_grad(...)`."""

    def value_and_grad[P1, R1, B1, L1](
        self,
        fun: Callable[[P1, R1, B1], L1],
        /,
    ) -> Callable[[P1, R1, B1], tuple[L1, P1]]:
        """Differentiate `fun(policy_params, ref_params, batch)` with respect to `policy_params`.

        Args:
            fun: Scalar loss callable.

        Returns:
            Callable returning `(loss, policy_grads)`.

        """
        ...


type _OptaxUpdatesTree = (
    PyTreeNode | Iterable[_OptaxUpdatesTree] | Mapping[str, _OptaxUpdatesTree] | complex | bytes | None
)


@runtime_checkable
class _OptaxApplyUpdatesProtocol(Protocol):
    """Protocol for `optax` providing `.apply_updates(params, updates)`."""

    def apply_updates[ParamsT](self, params: ParamsT, updates: _OptaxUpdatesTree, /) -> ParamsT:
        """Apply parameter updates.

        Args:
            params: Current parameter PyTree.
            updates: Optimizer updates PyTree.

        Returns:
            Updated parameter PyTree.

        """
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


@runtime_checkable
class _JnpDPOModuleProtocol(JnpZerosInt32Protocol, Protocol):
    """Protocol for `jax.numpy` exposing `int32`, `zeros`, and `sum` in MaxText DPO."""

    def sum(self, a: JaxArrayProtocol | float, /) -> SupportsFloat:
        """Sum elements of `a`.

        Args:
            a: Input array or scalar product.

        Returns:
            Scalar sum convertible via `float(...)`.

        """
        ...


@runtime_checkable
class _OptaxDPOModuleProtocol(Protocol):
    """Protocol for `optax` exposing `adamw` in MaxText DPO."""

    def adamw(
        self,
        learning_rate: float,
        /,
    ) -> _OptaxOptimizerProtocol[dict[str, JaxArrayProtocol], dict[str, JaxArrayProtocol], dict[str, PyTreeNode]]:
        """Construct an `adamw` gradient transformation.

        Args:
            learning_rate: Optimizer learning rate.

        Returns:
            Configured `_OptaxOptimizerProtocol`.

        """
        ...


jax: JaxDistributedAndPRNGProtocol | None = optional_import("jax", JaxDistributedAndPRNGProtocol)
jnn = optional_import("jax.nn")
jnp: _JnpDPOModuleProtocol | None = optional_import("jax.numpy", _JnpDPOModuleProtocol)
optax: _OptaxDPOModuleProtocol | None = optional_import("optax", _OptaxDPOModuleProtocol)
Gemma4Model: MaxTextModelConstructorProtocol | None = import_optional_symbol(
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


def _compute_logps[ValT: TensorType | ArithmeticTensorProtocol | float](
    model: MaxTextDPOModel[ValT],
    params: Mapping[str, ValT],
    inputs: ValT,
    labels: ValT,
) -> ValT | JaxSliceableTensorProtocol | float:
    """Compute sequence log probabilities from logits and target labels.

    Applies `jax.nn.log_softmax` over the vocabulary dimension, gathers shifted next-token
    log-probabilities via `gather_jax_sequence_logps`, masks out padding/prompt tokens,
    and sums along the sequence axis.

    Args:
        model: Stateless Flax/MaxText neural network model instance.
        params: Model parameters dictionary.
        inputs: Input token IDs tensor of shape `(batch_size, seq_len)`.
        labels: Target label IDs tensor of shape `(batch_size, seq_len)`.

    Returns:
        Summed non-padding log probability per sequence.

    """
    assert isinstance(model, _StatelessApplyModelProtocol)
    logits = model.apply(params, inputs)
    if (
        isinstance(jnn, JaxNNLogSoftmaxProtocol)
        and isinstance(jnp, JaxNumpyLogpOpsProtocol)
        and isinstance(logits, JaxSliceableTensorProtocol)
        and isinstance(labels, JaxSliceableTensorProtocol)
        and len(logits.shape) >= 2
    ):
        return gather_jax_sequence_logps(jnn, jnp, logits, labels)
    if jnp is not None and isinstance(logits, (int, float)) and isinstance(labels, (int, float)):
        sum_val: float = float(jnp.sum(logits * labels))
        return sum_val
    return logits


def _dpo_step_loss[ValT: TensorType | float](
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
    pi_ch_raw = _compute_logps(policy_model, policy_params, batch["chosen_inputs"], batch["chosen_labels"])
    pi_re_raw = _compute_logps(policy_model, policy_params, batch["rejected_inputs"], batch["rejected_labels"])
    ref_ch_raw = _compute_logps(ref_model, ref_params, batch["chosen_inputs"], batch["chosen_labels"])
    ref_re_raw = _compute_logps(ref_model, ref_params, batch["rejected_inputs"], batch["rejected_labels"])
    assert isinstance(pi_ch_raw, (ArithmeticTensorProtocol, float))
    assert isinstance(pi_re_raw, (ArithmeticTensorProtocol, float))
    assert isinstance(ref_ch_raw, (ArithmeticTensorProtocol, float))
    assert isinstance(ref_re_raw, (ArithmeticTensorProtocol, float))
    (loss, _, _) = dpo_loss(pi_ch_raw, pi_re_raw, ref_ch_raw, ref_re_raw, beta)
    return loss


def _get_train_step_fn[
    ValT: (TensorProtocol | TorchTensorProtocol | TfTensorProtocol | MlxArrayProtocol | float),
    UpdatesT: _OptaxUpdatesTree,
    OptStateT,
](
    policy_model: MaxTextDPOModel[ValT],
    ref_model: MaxTextDPOModel[ValT],
    optimizer: _OptaxOptimizerProtocol[dict[str, ValT], UpdatesT, OptStateT],
    beta: float,
) -> MaxTextDPOStepCallable[dict[str, ValT], OptStateT, Mapping[str, ValT]]:
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
        policy_params: dict[str, ValT],
        ref_params: dict[str, ValT],
        opt_state: OptStateT,
        batch: Mapping[str, ValT],
    ) -> tuple[
        dict[str, ValT],
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
            p: dict[str, ValT],
            r: dict[str, ValT],
            b: Mapping[str, ValT],
        ) -> ValT | ArithmeticTensorProtocol | float:
            """Compute DPO step loss for gradient differentiation.

            Args:
                p: Policy parameter mapping.
                r: Reference parameter mapping.
                b: Batch mapping.

            Returns:
                Scalar DPO step loss.

            """
            return _dpo_step_loss(policy_model, p, ref_model, r, b, beta)

        assert isinstance(jax, _JaxValueAndGrad3Protocol)
        assert isinstance(optax, _OptaxApplyUpdatesProtocol)
        (loss, grads) = jax.value_and_grad(_wrapper)(policy_params, ref_params, batch)
        (updates, new_opt_state) = optimizer.update(grads, opt_state, policy_params)
        updated_params: dict[str, ValT] = optax.apply_updates(policy_params, updates)
        scalar_loss: SupportsFloat | float = loss
        return (updated_params, new_opt_state, scalar_loss)

    if isinstance(jax, _JaxJitProtocol):
        return jax.jit(train_step)
    return train_step


def _run_training_epochs[ParamsT, OptT, BatchT](
    state: MaxTextDPOState[ParamsT, OptT, BatchT],
) -> tuple[ParamsT, OptT, float]:
    """Run DPO training epochs for MaxText.

    Args:
        state: `MaxTextDPOState` containing dataloader and parameters.

    Returns:
        A 3-tuple of `(policy_params, opt_state, final_loss)`.

    """
    policy_params = state.policy_params
    opt_state = state.opt_state
    step_fn = state.train_step
    dataloader: Iterable[BatchT] = state.dataloader

    def process_batch(batch: BatchT) -> float:
        """Process a single training batch.

        Args:
            batch: Dictionary containing training batch tensors.

        Returns:
            Computed batch loss scalar.

        """
        nonlocal policy_params, opt_state
        policy_params, opt_state, loss = step_fn(
            policy_params,
            state.ref_params,
            opt_state,
            batch,
        )
        return extract_scalar_float(loss)

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
    policy_params: dict[str, JaxArrayProtocol] = {
        str(k): v for k, v in dict(policy_model.init(rng, dummy_input)).items() if isinstance(v, JaxArrayProtocol)
    }
    ref_params: dict[str, JaxArrayProtocol] = {
        str(k): v for k, v in dict(ref_model.init(rng, dummy_input)).items() if isinstance(v, JaxArrayProtocol)
    }
    optimizer: _OptaxOptimizerProtocol[
        dict[str, JaxArrayProtocol],
        dict[str, JaxArrayProtocol],
        dict[str, PyTreeNode],
    ] = optax.adamw(learning_rate)
    opt_state = optimizer.init(policy_params)
    train_step: MaxTextDPOStepCallable[
        dict[str, JaxArrayProtocol],
        dict[str, PyTreeNode],
        Mapping[str, JaxArrayProtocol],
    ] = _get_train_step_fn(policy_model, ref_model, optimizer, beta)
    data_dict = build_dataloader(ETLConfig(dataset_name=dataset, split="train", batch_size=batch_size))
    dataloader = data_dict.get("loader", None)
    has_loader = isinstance(dataloader, Iterable)

    if not has_loader:
        raise ValueError(f"Invalid dataloader for dataset: {dataset}")

    (policy_params, opt_state, final_loss) = _run_training_epochs(
        MaxTextDPOState(
            dataloader=dataloader,
            epochs=epochs,
            train_step=train_step,
            policy_params=policy_params,
            ref_params=ref_params,
            opt_state=opt_state,
            beta=beta,
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
    from gemma_4_sql.backends._jax_common import (
        JaxShardingDevicesModuleProtocol,
        build_tpu_mesh_and_sharding,
        initialize_jax_distributed_if_configured,
    )

    _ = initialize_jax_distributed_if_configured(jax)
    if bool(kwargs.get("strict_hardware", False)):
        _ = build_tpu_mesh_and_sharding(
            jax if isinstance(jax, JaxShardingDevicesModuleProtocol) else None,
            str(kwargs.get("hardware", "cpu")),
            strict_hardware=True,
            backend_label="maxtext",
        )
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

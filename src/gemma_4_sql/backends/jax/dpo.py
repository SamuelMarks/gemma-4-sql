"""JAX-specific DPO (Direct Preference Optimization) logic."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Protocol, SupportsFloat, runtime_checkable

import jax
import jax.nn as jnn
import jax.numpy as jnp
import optax
from flax import nnx
from jax import Array

from gemma_4_sql.backends.common_dpo import (
    ArithmeticTensorProtocol,
    ScalarItemProtocol,
    gather_jax_sequence_logps,
    generic_dpo_loss,
    run_dpo_wrapper,
)
from gemma_4_sql.backends.common_inference import call_jax_causal_lm
from gemma_4_sql.backends.common_train import (
    NNXOptimizerProtocol,
    create_nnx_optimizer,
    generic_run_training_epochs,
)
from gemma_4_sql.backends.common_train import (
    apply_nnx_optimizer_update as _apply_nnx_optimizer_update,
)
from gemma_4_sql.backends.jax.etl import build_dataloader
from gemma_4_sql.backends.jax.gemma4.config import ModelConfig, resolve_gemma4_config
from gemma_4_sql.backends.jax.gemma4.modeling import Gemma4ForCausalLM
from gemma_4_sql.backends.jax.inference import MODEL_CACHE as _MODEL_CACHE
from gemma_4_sql.exceptions import DependencyMissingError, StatusValue
from gemma_4_sql.type_hints import DPOConfig, DPOLoopState, DPOResultDict, ETLConfig, JaxCausalLMProtocol, TensorType, TrainerState

Gemma4Config = ModelConfig
type JaxDPOBatch = Mapping[str, Array | TensorType | Sequence[int]]
type JaxDPOStepFn = Callable[..., Array | SupportsFloat | float]


@runtime_checkable
class _NNXValueAndGradModuleProtocol(Protocol):
    """Protocol for a `flax.nnx` module exposing `value_and_grad` and `jit`."""

    def value_and_grad(self, fn: Callable[..., Array], /) -> Callable[..., tuple[Array, Array]]:
        """Create a value-and-gradient function for an NNX module."""
        ...

    def jit(self, fn: JaxDPOStepFn, /) -> JaxDPOStepFn:
        """JIT-compile an NNX function."""
        ...


def _create_nnx_optimizer[ModelT: nnx.Module](
    model: ModelT,
    tx: optax.GradientTransformation,
) -> nnx.Optimizer[ModelT]:
    """Create a Flax NNX Optimizer (`nnx.Optimizer(model, tx, wrt=nnx.Param)`).

    Args:
        model: Flax NNX module to optimize.
        tx: Optax gradient transformation.

    Returns:
        Configured `nnx.Optimizer` instance.

    """
    res: nnx.Optimizer[ModelT] = create_nnx_optimizer(nnx, model, tx)  # pyright: ignore[reportAssignmentType]
    return res


def dpo_loss[TensorT: ArithmeticTensorProtocol | float](
    policy_chosen_logps: TensorT,
    policy_rejected_logps: TensorT,
    ref_chosen_logps: TensorT,
    ref_rejected_logps: TensorT,
    beta: float = 0.1,
) -> tuple[TensorT | ArithmeticTensorProtocol | float, TensorT, TensorT]:
    """Compute the Direct Preference Optimization (DPO) loss for JAX arrays.

    Args:
        policy_chosen_logps: Log probabilities of the chosen completions from the policy model.
        policy_rejected_logps: Log probabilities of the rejected completions from the policy model.
        ref_chosen_logps: Log probabilities of the chosen completions from the reference model.
        ref_rejected_logps: Log probabilities of the rejected completions from the reference model.
        beta: The beta parameter controlling the KL penalty.

    Returns:
        A 3-tuple of `(mean_loss, chosen_rewards, rejected_rewards)`.

    Raises:
        DependencyMissingError: If JAX DPO dependencies are missing.

    """
    if jnp is None or jnn is None:
        raise DependencyMissingError("JAX DPO dependencies are missing.", backend="jax")
    return generic_dpo_loss(  # pyright: ignore[reportReturnType]  # ty: ignore[invalid-return-type]
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        beta,
        jnn.log_sigmoid,
    )


def _compute_logps(
    model: Callable[..., tuple[Array, ...] | Array] | JaxCausalLMProtocol,
    inputs: Array,
    labels: Array,
) -> Array:
    """Compute sequence log probabilities for DPO using token-level log-softmax.

    Args:
        model: Policy or reference model callable.
        inputs: Input token ID tensor of shape `(batch_size, seq_len)`.
        labels: Target token ID tensor of shape `(batch_size, seq_len)`.

    Returns:
        Summed non-padding token log probabilities per sequence.

    """
    seq_len = inputs.shape[1] if len(inputs.shape) > 1 else 1
    positions = jnp.arange(seq_len)[None, :]
    logits: Array = call_jax_causal_lm(model, inputs, positions)  # pyright: ignore[reportArgumentType,reportAssignmentType]  # ty: ignore[invalid-assignment]
    res: Array = gather_jax_sequence_logps(jnn, jnp, logits, labels)
    return res


def _dpo_step_loss(
    policy_model: Callable[..., tuple[Array, ...] | Array] | JaxCausalLMProtocol,
    ref_model: Callable[..., tuple[Array, ...] | Array] | JaxCausalLMProtocol,
    batch: Mapping[str, Array],
    beta: float,
) -> Array:
    """Compute scalar DPO loss for a preference batch.

    Args:
        policy_model: Trainable policy model.
        ref_model: Frozen reference model.
        batch: Preference batch dictionary with chosen/rejected inputs and labels.
        beta: KL penalty temperature coefficient.

    Returns:
        Scalar DPO loss array.

    """
    ch_inputs = batch["chosen_inputs"] if "chosen_inputs" in batch else batch["chosen_input_ids"]
    re_inputs = batch["rejected_inputs"] if "rejected_inputs" in batch else batch["rejected_input_ids"]
    ch_labels = batch["chosen_labels"]
    re_labels = batch["rejected_labels"]
    pi_ch_logps = _compute_logps(policy_model, ch_inputs, ch_labels)
    pi_re_logps = _compute_logps(policy_model, re_inputs, re_labels)
    ref_ch_logps = _compute_logps(ref_model, ch_inputs, ch_labels)
    ref_re_logps = _compute_logps(ref_model, re_inputs, re_labels)
    (loss, _, _) = dpo_loss(pi_ch_logps, pi_re_logps, ref_ch_logps, ref_re_logps, beta)
    return jnp.asarray(loss)


def _get_train_step_fn(beta: float) -> JaxDPOStepFn:
    """Return a JIT-compiled DPO training step function for the given beta.

    Args:
        beta: KL penalty temperature coefficient.

    Returns:
        Compiled training step callable.

    Raises:
        DependencyMissingError: If `flax.nnx.value_and_grad` is unavailable.

    """
    if nnx is None or not isinstance(nnx, _NNXValueAndGradModuleProtocol):
        raise DependencyMissingError(
            "Flax NNX value_and_grad is required for JAX DPO.",
            dependency_name="flax.nnx",
            backend="jax",
        )

    def train_step(
        policy_model: JaxCausalLMProtocol,
        ref_model: JaxCausalLMProtocol,
        optimizer: NNXOptimizerProtocol[JaxCausalLMProtocol, Array],
        batch: Mapping[str, Array],
    ) -> Array:
        """Execute a single JAX-compiled DPO training step.

        Args:
            policy_model: Trainable policy model.
            ref_model: Frozen reference model.
            optimizer: Flax NNX optimizer instance.
            batch: Preference batch dictionary.

        Returns:
            Scalar step loss.

        """

        def _loss_wrapper(
            p: JaxCausalLMProtocol,
            r: JaxCausalLMProtocol,
            b: Mapping[str, Array],
        ) -> Array:
            """Evaluate `_dpo_step_loss` for gradient transformation."""
            return _dpo_step_loss(p, r, b, beta)

        (loss, grads) = nnx.value_and_grad(_loss_wrapper)(policy_model, ref_model, batch)
        _apply_nnx_optimizer_update(optimizer, policy_model, grads)
        return loss

    return nnx.jit(train_step)


def _run_training_epochs[ModelT, OptT, BatchT, ParamsT](
    state: DPOLoopState[ModelT, OptT, BatchT] | TrainerState[ModelT, ParamsT, OptT, BatchT],
) -> float:
    """Run DPO training epochs over the preference dataloader.

    Args:
        state: `DPOLoopState` holding models, optimizer, dataloader, and step function.

    Returns:
        Mean batch loss from the final epoch.

    """
    dataloader: Iterable[BatchT] = state.dataloader or ()
    policy_model = state.policy_model
    ref_model = state.ref_model
    optimizer = state.optimizer
    train_step = state.train_step

    def _step(batch: BatchT) -> float:
        """Execute a single JAX DPO step and extract scalar float loss."""
        assert train_step is not None
        loss = train_step(policy_model, ref_model, optimizer, batch)
        if isinstance(loss, ScalarItemProtocol):
            return float(loss.item())
        if isinstance(loss, (int, float, SupportsFloat)):
            return float(loss)
        return 0.0

    return generic_run_training_epochs(state.epochs, dataloader, _step)


def _resolve_gemma4_config(model_name: str) -> ModelConfig:
    """Resolve a `Gemma4Config` preset for the given `model_name`.

    Args:
        model_name: Model name or preset identifier.

    Returns:
        Resolved `ModelConfig` object.

    """
    return resolve_gemma4_config(Gemma4Config, model_name)


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
        model_name: The name of the model.
        dataset: The dataset name.
        beta: The beta temperature parameter.
        epochs: Number of training epochs.
        learning_rate: Learning rate for the optimizer.
        batch_size: Batch size for training.

    Returns:
        A tuple of `(status, final_loss)`.

    Raises:
        DependencyMissingError: If JAX dependencies are missing.
        ValueError: If dataloader is invalid.

    """
    if jax is None or jnp is None or optax is None or Gemma4ForCausalLM is None or nnx is None:  # pyright: ignore[reportUnnecessaryComparison]
        raise DependencyMissingError("JAX dependencies are missing for DPO.")

    resolved_cfg = _resolve_gemma4_config(model_name)
    policy_model = Gemma4ForCausalLM(resolved_cfg, rngs=nnx.Rngs(0))
    ref_model = Gemma4ForCausalLM(resolved_cfg, rngs=nnx.Rngs(1))
    optimizer = _create_nnx_optimizer(policy_model, optax.adamw(learning_rate))
    train_step = _get_train_step_fn(beta)
    data_dict = build_dataloader(ETLConfig(dataset_name=dataset, split="train", batch_size=batch_size))
    dataloader = data_dict.get("loader", None)
    has_loader = isinstance(dataloader, Iterable)

    if not has_loader:
        raise ValueError(f"Invalid dataloader for dataset: {dataset}")

    final_loss = _run_training_epochs(
        DPOLoopState(
            dataloader=dataloader,
            epochs=epochs,
            policy_model=policy_model,
            ref_model=ref_model,
            optimizer=optimizer,
            train_step=train_step,
        )
    )
    _MODEL_CACHE.put_model(model_name, policy_model)
    return "completed", final_loss


def run_dpo(config: DPOConfig, **kwargs: StatusValue) -> DPOResultDict:
    """Run a DPO training loop for JAX.

    Args:
        config: DPO training configuration.
        **kwargs: Hyperparameters for DPO (e.g., `beta`, `learning_rate`, `raise_on_error`).

    Returns:
        Structured `DPOResultDict` with execution status and metrics.

    Raises:
        DependencyMissingError: If JAX dependencies are missing.
        TrainingError: If DPO fails and `raise_on_error` is enabled.

    """
    if jax is None or jnp is None or jnn is None or optax is None or Gemma4ForCausalLM is None:  # pyright: ignore[reportUnnecessaryComparison]
        raise DependencyMissingError("JAX DPO dependencies are missing.", backend="jax")

    return run_dpo_wrapper(
        "jax",
        config,
        lambda m, d, b, ep, lr, bs: _execute_dpo(m, d, b, ep, lr, batch_size=bs),
        **kwargs,
    )

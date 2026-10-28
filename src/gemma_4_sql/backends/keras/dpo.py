"""Keras-specific DPO (Direct Preference Optimization) logic."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Protocol, SupportsFloat, runtime_checkable

from gemma_4_sql.backends.common_dpo import ArithmeticTensorProtocol, generic_dpo_loss, run_dpo_wrapper
from gemma_4_sql.backends.common_train import generic_run_training_epochs
from gemma_4_sql.backends.keras.etl import build_dataloader
from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.exceptions import DependencyMissingError, StatusValue
from gemma_4_sql.type_hints import (
    DPOConfig,
    DPOResultDict,
    ETLConfig,
    KerasCausalLMProtocol,
    KerasTrainerState,
    KerasVariableProtocol,
    TfTensorProtocol,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class _GemmaCausalLMLoaderProtocol(Protocol):
    """Protocol for `keras_nlp.models.GemmaCausalLM` exposing `.from_preset(model_name)`."""

    def from_preset(self, model_name: str, /) -> KerasCausalLMProtocol:
        """Load a `GemmaCausalLM` model from a preset identifier."""
        ...


@runtime_checkable
class _KerasOptimizerProtocol[TensorT, VarT](Protocol):
    """Protocol for a Keras optimizer supporting `.apply_gradients(...)`."""

    def apply_gradients(
        self,
        grads_and_vars: Iterable[tuple[TensorT, VarT]],
        /,
    ) -> None:
        """Apply gradients to variables."""
        ...


tf = optional_import("tensorflow")
keras = optional_import("keras")
GemmaCausalLM: _GemmaCausalLMLoaderProtocol | None = optional_attr(
    "keras_nlp.models",
    "GemmaCausalLM",
    _GemmaCausalLMLoaderProtocol,
)


def dpo_loss[TensorT: ArithmeticTensorProtocol | float](
    policy_chosen_logps: TensorT,
    policy_rejected_logps: TensorT,
    ref_chosen_logps: TensorT,
    ref_rejected_logps: TensorT,
    beta: float = 0.1,
) -> tuple[TensorT | ArithmeticTensorProtocol | float, TensorT, TensorT]:
    """Compute the Direct Preference Optimization (DPO) loss for TensorFlow/Keras tensors.

    Args:
        policy_chosen_logps: Log probabilities of the chosen completions from the policy model.
        policy_rejected_logps: Log probabilities of the rejected completions from the policy model.
        ref_chosen_logps: Log probabilities of the chosen completions from the reference model.
        ref_rejected_logps: Log probabilities of the rejected completions from the reference model.
        beta: The beta parameter controlling the KL penalty.

    Returns:
        A 3-tuple of `(mean_loss, chosen_rewards, rejected_rewards)`.

    Raises:
        DependencyMissingError: If TensorFlow is missing.

    """
    if tf is None:
        raise DependencyMissingError("Keras DPO dependencies are missing.", backend="keras")
    (losses, chosen_rewards, rejected_rewards) = generic_dpo_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        beta,
        tf.math.log_sigmoid,
    )
    mean_loss: TensorT | ArithmeticTensorProtocol | float = losses
    with contextlib.suppress(AttributeError, TypeError):
        mean_loss = tf.reduce_mean(losses)
    return (mean_loss, chosen_rewards, rejected_rewards)


type KerasDPOModel[TensorT] = Callable[[TensorT], TensorT] | KerasCausalLMProtocol


def _compute_logps[TensorT: TfTensorProtocol | SupportsFloat](
    model: KerasDPOModel[TensorT],
    inputs: TensorT,
    labels: TensorT,
) -> TensorT:
    """Compute sequence log probabilities for DPO using TensorFlow log-softmax.

    Args:
        model: Keras causal LM callable.
        inputs: Input token ID tensor of shape `(batch_size, seq_len)`.
        labels: Target token ID tensor of shape `(batch_size, seq_len)`.

    Returns:
        Summed non-padding token log probabilities per sequence.

    Raises:
        DependencyMissingError: If TensorFlow is not installed.

    """
    if tf is None:
        raise DependencyMissingError("Keras DPO dependencies are missing.", backend="keras")
    tf_cast = tf.dtypes.cast
    logits = model(inputs)  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    labels_expanded = tf.expand_dims(tf_cast(labels, tf.int32), axis=-1)
    selected_log_probs = tf.gather(log_probs, labels_expanded, batch_dims=2)
    selected_log_probs = tf.squeeze(selected_log_probs, axis=-1)
    mask = tf_cast(labels != 0, tf.float32)
    res: TensorT = tf.reduce_sum(selected_log_probs * mask, axis=-1)
    return res


def _get_train_step_fn[TensorT: ArithmeticTensorProtocol, VarT](
    policy_model: KerasDPOModel[TensorT],
    ref_model: KerasDPOModel[TensorT],
    optimizer: _KerasOptimizerProtocol[TensorT, VarT],
    beta: float,
) -> Callable[[Mapping[str, TensorT]], TensorT]:
    """Return a `tf.function` compiled DPO train step function.

    Args:
        policy_model: Trainable policy model.
        ref_model: Frozen reference model.
        optimizer: Keras optimizer instance.
        beta: KL penalty temperature coefficient.

    Returns:
        Compiled TensorFlow step function.

    Raises:
        DependencyMissingError: If TensorFlow is not installed.

    """
    if tf is None:
        raise DependencyMissingError("Keras DPO dependencies are missing.", backend="keras")

    def train_step(
        batch: Mapping[str, TensorT],
    ) -> TensorT:
        """Compute DPO loss and apply gradients under a TensorFlow GradientTape.

        Args:
            batch: Preference batch dictionary.

        Returns:
            Scalar step loss tensor.

        """
        assert tf is not None
        with tf.GradientTape() as tape:
            pi_ch_logps = _compute_logps(policy_model, batch["chosen_inputs"], batch["chosen_labels"])
            pi_re_logps = _compute_logps(policy_model, batch["rejected_inputs"], batch["rejected_labels"])
            ref_ch_logps = _compute_logps(ref_model, batch["chosen_inputs"], batch["chosen_labels"])
            ref_re_logps = _compute_logps(ref_model, batch["rejected_inputs"], batch["rejected_labels"])
            (loss, _, _) = dpo_loss(pi_ch_logps, pi_re_logps, ref_ch_logps, ref_re_logps, beta)
        trainable_vars: Sequence[KerasVariableProtocol | TfTensorProtocol | TensorT] = ()
        with contextlib.suppress(AttributeError):
            trainable_vars = policy_model.trainable_variables  # pyright: ignore[reportAttributeAccessIssue,reportFunctionMemberAccess]  # ty: ignore[unresolved-attribute]
        grads = tape.gradient(loss, trainable_vars)
        optimizer.apply_gradients(zip(grads, trainable_vars))  # pyright: ignore[reportAttributeAccessIssue,reportArgumentType]  # ty: ignore[invalid-argument-type]
        return loss  # pyright: ignore[reportReturnType]  # ty: ignore[invalid-return-type]

    compiled_fn: Callable[[Mapping[str, TensorT]], TensorT] = tf.function(train_step)
    return compiled_fn


def _run_training_epochs[ModelT, OptT, BatchT](
    state: KerasTrainerState[ModelT, OptT, BatchT],
) -> float:
    """Run Keras DPO training epochs over the preference dataloader.

    Args:
        state: `KerasTrainerState` holding dataloader, epoch count, and compiled step function.

    Returns:
        Mean batch loss from the final epoch.

    """
    dataloader: Iterable[BatchT] = state.dataloader or ()
    train_step = state.train_step

    def _step(batch: BatchT) -> float:
        """Execute a single Keras DPO step and extract scalar float loss."""
        assert train_step is not None
        loss = train_step(batch)
        if isinstance(loss, (int, float, SupportsFloat)):
            return float(loss)
        return 0.0

    return generic_run_training_epochs(state.epochs, dataloader, _step)


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
        model_name: Target model name.
        dataset: Target dataset name.
        beta: DPO beta temperature.
        epochs: Number of training epochs.
        learning_rate: Learning rate for optimizer.
        batch_size: Training batch size.

    Returns:
        A tuple of `(status, final_loss)`.

    Raises:
        DependencyMissingError: If Keras dependencies are missing.
        ValueError: If model loading or dataloader fails.

    """
    if keras is None or tf is None:
        raise DependencyMissingError("Keras dependencies are missing.")

    try:
        gemma_causal_lm_cls = GemmaCausalLM if GemmaCausalLM is not None else optional_attr("keras_nlp.models", "GemmaCausalLM", _GemmaCausalLMLoaderProtocol)
        if gemma_causal_lm_cls is None:
            raise ImportError("keras_nlp.models.GemmaCausalLM is not available.")
        policy_model: KerasCausalLMProtocol = gemma_causal_lm_cls.from_preset(model_name)
        ref_model: KerasCausalLMProtocol = gemma_causal_lm_cls.from_preset(model_name)
    except (ImportError, ValueError) as e:
        raise ValueError(f"Failed to load Keras model {model_name}") from e

    optimizer: _KerasOptimizerProtocol[TfTensorProtocol, KerasVariableProtocol] = keras.optimizers.AdamW(learning_rate=learning_rate)
    train_step = _get_train_step_fn(policy_model, ref_model, optimizer, beta)
    data_dict = build_dataloader(ETLConfig(dataset_name=dataset, split="train", batch_size=batch_size))
    dataloader = data_dict.get("loader", None)
    has_loader = isinstance(dataloader, Iterable)

    if not has_loader:
        raise ValueError(f"Invalid dataloader for dataset: {dataset}")

    final_loss = _run_training_epochs(KerasTrainerState(dataloader=dataloader, epochs=epochs, train_step=train_step))
    return "completed", final_loss


def run_dpo(config: DPOConfig, **kwargs: StatusValue) -> DPOResultDict:
    """Run a DPO training loop for Keras.

    Args:
        config: DPO configuration object.
        **kwargs: Hyperparameters for DPO (e.g., `beta`, `learning_rate`, `raise_on_error`).

    Returns:
        Structured `DPOResultDict` containing execution status and loss.

    Raises:
        DependencyMissingError: If Keras DPO dependencies are missing.
        TrainingError: If DPO fails and `raise_on_error` is enabled.

    """
    if keras is None or tf is None:
        raise DependencyMissingError("Keras DPO dependencies are missing.", backend="keras")

    return run_dpo_wrapper(
        "keras",
        config,
        lambda m, d, b, ep, lr, bs: _execute_dpo(m, d, b, ep, lr, batch_size=bs),
        **kwargs,
    )

"""Keras-specific DPO (Direct Preference Optimization) logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from gemma_4_sql.backends.common_dpo import generic_dpo_loss
from gemma_4_sql.backends.keras.etl import build_dataloader
from gemma_4_sql.backends.lazy_loader import optional_attr, optional_import
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import DPOConfig, ETLConfig, ModelType, TensorType, TrainerState

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import DPOResultDict
logger = logging.getLogger(__name__)

tf = optional_import("tensorflow")
keras = optional_import("keras")
GemmaCausalLM = optional_attr("keras_nlp.models", "GemmaCausalLM")


def dpo_loss(policy_chosen_logps: TensorType, policy_rejected_logps: TensorType, ref_chosen_logps: TensorType, ref_rejected_logps: TensorType, beta: float = 0.1) -> tuple[Any, Any, Any]:
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
    if tf is None:
        return (0.0, 0.0, 0.0)
    return generic_dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta, tf.math.log_sigmoid)


def _compute_logps(model: ModelType, inputs: TensorType, labels: TensorType) -> Any:
    """Compute sequence log probabilities for DPO using TensorFlow log-softmax.

    Args:
        model: Keras causal LM callable.
        inputs: Input token ID tensor.
        labels: Target token ID tensor.

    Returns:
        Summed non-padding token log probabilities per sequence.

    """
    if tf is None:
        return 0.0
    logits = model(inputs)

    # Compute log_softmax over the vocabulary dimension (usually axis -1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    # Gather the log probability of the true next token (label).
    # labels shape is (batch_size, sequence_length)
    # log_probs shape is (batch_size, sequence_length, vocab_size)
    labels_expanded = tf.expand_dims(tf.cast(labels, tf.int32), axis=-1)
    selected_log_probs = tf.gather(log_probs, labels_expanded, batch_dims=2)

    # Remove the extra dimension and sum over the sequence length
    selected_log_probs = tf.squeeze(selected_log_probs, axis=-1)
    # We might want to mask out padding tokens in the future, assuming non-zero labels are valid tokens for now
    mask = tf.cast(cast(Any, labels) != 0, tf.float32)  # Justified: Dynamic backend protocol typing
    return tf.reduce_sum(selected_log_probs * mask, axis=-1)


def _get_train_step_fn(policy_model: ModelType, ref_model: ModelType, optimizer: object, beta: float) -> object:
    """Return a `tf.function` compiled DPO train step function.

    Args:
        policy_model: Trainable policy model.
        ref_model: Frozen reference model.
        optimizer: Keras optimizer instance.
        beta: KL penalty temperature coefficient.

    Returns:
        Compiled TensorFlow step function or no-op fallback if TensorFlow is unavailable.

    """
    if tf is None:

        def _dummy_step(_b: Any) -> float:
            """Return 0.0 when TensorFlow is not installed."""
            return 0.0

        return _dummy_step

    def train_step(batch: dict[str, object]) -> object:
        """Compute DPO loss and apply gradients under a TensorFlow GradientTape.

        Args:
            batch: Preference batch dictionary.

        Returns:
            Scalar step loss tensor.

        """
        with tf.GradientTape() as tape:
            pi_ch_logps = _compute_logps(policy_model, batch["chosen_inputs"], batch["chosen_labels"])  # type: ignore # Justified: Dynamic backend protocol typing
            pi_re_logps = _compute_logps(policy_model, batch["rejected_inputs"], batch["rejected_labels"])  # type: ignore # Justified: Dynamic backend protocol typing
            ref_ch_logps = _compute_logps(ref_model, batch["chosen_inputs"], batch["chosen_labels"])  # type: ignore # Justified: Dynamic backend protocol typing
            ref_re_logps = _compute_logps(ref_model, batch["rejected_inputs"], batch["rejected_labels"])  # type: ignore # Justified: Dynamic backend protocol typing
            (loss, _, _) = dpo_loss(pi_ch_logps, pi_re_logps, ref_ch_logps, ref_re_logps, beta)
        grads = tape.gradient(loss, getattr(policy_model, "trainable_variables", []))
        optimizer.apply_gradients(zip(grads, getattr(policy_model, "trainable_variables", [])))  # type: ignore # Justified: Dynamic backend protocol typing
        return loss

    return tf.function(train_step)


def _run_training_epochs(state: TrainerState) -> float:
    """Run Keras DPO training epochs over the preference dataloader.

    Args:
        state: `TrainerState` holding dataloader, epoch count, and compiled step function.

    Returns:
        Mean batch loss from the final epoch.

    """
    dataloader = state.dataloader
    epochs = state.epochs
    train_step = state.train_step
    final_loss = 0.0
    for _epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            loss: Any = train_step(batch)
            if hasattr(loss, "numpy"):
                epoch_loss += float(loss.numpy())
            elif isinstance(loss, (int, float)):
                epoch_loss += float(loss)
        final_loss = epoch_loss / max(1, len(dataloader) if hasattr(dataloader, "__len__") else 1)
    return float(final_loss)


def _execute_dpo(model_name: str, dataset: str, beta: float, epochs: int, learning_rate: float, batch_size: int = 2) -> tuple[str, float]:
    """Execute the core DPO loop.

    Args:
        model_name: Target model name.
        dataset: Target dataset name.
        beta: DPO beta temperature.
        epochs: Number of training epochs.
        learning_rate: Learning rate for optimizer.
        batch_size: Training batch size.

    Returns:
        A tuple of (status, final_loss).

    Raises:
        DependencyMissingError: If Keras dependencies are missing.
        ValueError: If model loading or dataloader fails.

    """
    if keras is None or tf is None:
        raise DependencyMissingError("Keras dependencies are missing.")

    try:
        gemma_causal_lm_cls = GemmaCausalLM if GemmaCausalLM is not None else optional_attr("keras_nlp.models", "GemmaCausalLM")
        if gemma_causal_lm_cls is None:
            raise ImportError("keras_nlp.models.GemmaCausalLM is not available.")
        policy_model: ModelType = gemma_causal_lm_cls.from_preset(model_name)
        ref_model: ModelType = gemma_causal_lm_cls.from_preset(model_name)
    except (ImportError, ValueError) as e:
        raise ValueError(f"Failed to load Keras model {model_name}") from e

    optimizer = keras.optimizers.AdamW(learning_rate=learning_rate)
    train_step = _get_train_step_fn(policy_model, ref_model, optimizer, beta)
    data_dict = build_dataloader(ETLConfig(dataset_name=dataset, split="train", batch_size=batch_size))
    dataloader = data_dict.get("loader", None)

    if dataloader is None or not hasattr(dataloader, "__iter__"):
        raise ValueError(f"Invalid dataloader for dataset: {dataset}")

    final_loss = _run_training_epochs(TrainerState(dataloader=dataloader, epochs=epochs, train_step=train_step))
    return "completed", final_loss


def run_dpo(config: DPOConfig, **kwargs: object) -> DPOResultDict:
    """Run a DPO training loop for Keras.

    Args:
        config: DPO configuration object.
        **kwargs: Hyperparameters for DPO (e.g., beta, learning_rate).

    Returns:
        A dictionary containing execution status and loss.

    Raises:
        DependencyMissingError: If Keras DPO dependencies are missing.

    """
    model_name = getattr(config, "model_name", "model")
    dataset = getattr(config, "dataset", "dataset")
    beta = getattr(config, "beta", 0.1)
    epochs = getattr(config, "epochs", 1)
    learning_rate = getattr(config, "learning_rate", 1e-05)
    final_loss = 0.0
    status = "completed"
    if keras is None or tf is None:
        raise DependencyMissingError("Keras DPO dependencies are missing.")

    batch_size = getattr(config, "batch_size", 2)
    try:
        status, final_loss = _execute_dpo(model_name, dataset, beta, epochs, learning_rate, batch_size=batch_size)
    except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError) as e:
        logger.exception("Keras DPO error: ")
        status = f"failed: {e!s}"

    return {"backend": "keras", "action": "dpo", "model": model_name, "dataset": dataset, "beta": beta, "status": status, "final_loss": final_loss}

"""Keras-specific DPO (Direct Preference Optimization) logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from gemma_4_sql.backends.common_dpo import generic_dpo_loss
from gemma_4_sql.backends.keras.etl import build_dataloader
from gemma_4_sql.backends.lazy_loader import catch_optional_imports
from gemma_4_sql.type_hints import DPOConfig, ETLConfig, TensorType, TrainerState

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict
logger = logging.getLogger(__name__)
keras = None
tf = None
with catch_optional_imports():
    import keras
    import tensorflow as tf


def dpo_loss(policy_chosen_logps: TensorType, policy_rejected_logps: TensorType, ref_chosen_logps: TensorType, ref_rejected_logps: TensorType, beta: float = 0.1) -> tuple[TensorType, TensorType, TensorType]:
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


def _compute_logps(model: keras.Model, inputs: keras.KerasTensor | tf.Tensor, labels: object) -> object:
    """Compute exact log probabilities for DPO math using categorical cross-entropy approach.

    Returns:
        The resulting output from the operation.

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
    mask = tf.cast(labels != 0, tf.float32)
    return tf.reduce_sum(selected_log_probs * mask, axis=-1)


def _get_train_step_fn(policy_model: object, ref_model: object, optimizer: object, beta: float) -> object:
    """Return a tf.function compiled train step function.

    Returns:
        object: The resulting output from the operation.

    """
    if tf is None:
        return lambda _b: 0.0

    def train_step(batch: dict[str, object]) -> object:
        """Execute the train step operation.

        Returns:
            object: The resulting output from the operation.

        """
        with tf.GradientTape() as tape:
            pi_ch_logps = _compute_logps(policy_model, batch["chosen_inputs"], batch["chosen_labels"])
            pi_re_logps = _compute_logps(policy_model, batch["rejected_inputs"], batch["rejected_labels"])
            ref_ch_logps = _compute_logps(ref_model, batch["chosen_inputs"], batch["chosen_labels"])
            ref_re_logps = _compute_logps(ref_model, batch["rejected_inputs"], batch["rejected_labels"])
            (loss, _, _) = dpo_loss(pi_ch_logps, pi_re_logps, ref_ch_logps, ref_re_logps, beta)
        grads = tape.gradient(loss, getattr(policy_model, "trainable_variables", []))
        optimizer.apply_gradients(zip(grads, getattr(policy_model, "trainable_variables", [])))
        return loss

    return tf.function(train_step)


def _run_training_epochs(state: TrainerState) -> float:
    """Execute function.

    Returns:
        The execution result.

    """
    dataloader = state.dataloader
    epochs = state.epochs
    train_step = state.train_step
    """Run training epochs.

    Returns:
        object: The resulting output from the operation.

    """
    final_loss = 0.0
    for _epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            loss = train_step(batch)
            if hasattr(loss, "numpy") and callable(loss.numpy):
                epoch_loss += float(loss.numpy())
            else:
                epoch_loss += float(loss)
        final_loss = epoch_loss / max(1, len(dataloader) if hasattr(dataloader, "__len__") else 1)
    return float(final_loss)


def _execute_dpo(model_name: str, dataset: str, beta: float, epochs: int, learning_rate: float) -> tuple[str, float]:
    """Execute the core DPO loop."""
    try:
        gemma_causal_lm_cls = __import__("keras_nlp.models", fromlist=["GemmaCausalLM"]).GemmaCausalLM
        policy_model = gemma_causal_lm_cls.from_preset(model_name)
        ref_model = gemma_causal_lm_cls.from_preset(model_name)
    except (ImportError, ValueError) as e:
        raise ValueError(f"Failed to load Keras model {model_name}") from e

    optimizer = keras.optimizers.AdamW(learning_rate=learning_rate)
    train_step = _get_train_step_fn(policy_model, ref_model, optimizer, beta)
    data_dict = build_dataloader(ETLConfig(dataset_name=dataset, split="train", batch_size=2))
    dataloader = data_dict.get("loader", None)

    if dataloader is None or not hasattr(dataloader, "__iter__"):
        raise ValueError(f"Invalid dataloader for dataset: {dataset}")

    final_loss = _run_training_epochs(TrainerState(dataloader=dataloader, epochs=epochs, train_step=train_step))
    return "completed", final_loss


def run_dpo(config: DPOConfig, **kwargs: object) -> JSONDict:
    """Execute function.


    Args:
        **kwargs: Hyperparameters for DPO (e.g., beta, learning_rate).
    Returns:
        The execution result.

    """
    model_name = getattr(config, "model_name", "model")
    dataset = getattr(config, "dataset", "dataset")
    beta = getattr(config, "beta", 0.1)
    epochs = getattr(config, "epochs", 1)
    learning_rate = getattr(config, "learning_rate", 1e-05)
    """Run a DPO training loop for Keras.

    Returns:
        object: The resulting output from the operation.

    """
    final_loss = 0.0
    status = "completed"
    if keras is None or tf is None:
        from gemma_4_sql.exceptions import DependencyMissingError

        raise DependencyMissingError("Keras DPO dependencies are missing.")

    try:
        status, final_loss = _execute_dpo(model_name, dataset, beta, epochs, learning_rate)
    except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError) as e:
        logger.exception("Keras DPO error: ")
        status = f"failed: {e!s}"

    return {"backend": "keras", "action": "dpo", "model": model_name, "dataset": dataset, "beta": beta, "status": status, "final_loss": final_loss}

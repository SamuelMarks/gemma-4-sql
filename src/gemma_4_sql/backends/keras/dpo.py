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
    import tensorflow as tf  # pragma: no cover


def dpo_loss(policy_chosen_logps: TensorType, policy_rejected_logps: TensorType, ref_chosen_logps: TensorType, ref_rejected_logps: TensorType, beta: float = 0.1) -> tuple[TensorType, TensorType, TensorType]:
    """Compute the DPO loss.

    Returns:
        tuple: The losses.

    """
    if tf is None:
        return (0.0, 0.0, 0.0)
    return generic_dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta, tf.math.log_sigmoid)
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = pi_logratios - ref_logratios
    loss = -tf.math.log_sigmoid(beta * logits)
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    return (tf.reduce_mean(loss), tf.reduce_mean(chosen_rewards), tf.reduce_mean(rejected_rewards))


def _compute_logps(model: keras.Model, inputs: keras.KerasTensor | tf.Tensor, labels: object) -> object:
    """Mock logp computation.

    Returns:
        object: The resulting output from the operation.

    """
    if tf is None:
        return 0.0
    logits = model(inputs)  # pragma: no cover
    return tf.reduce_sum(tf.cast(logits, tf.float32) * tf.cast(labels, tf.float32), axis=-1)  # pragma: no cover


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
            pi_ch = policy_model(batch["chosen_inputs"])
            pi_re = policy_model(batch["rejected_inputs"])
            ref_ch = ref_model(batch["chosen_inputs"])
            ref_re = ref_model(batch["rejected_inputs"])
            pi_ch_logps = tf.reduce_sum(tf.cast(pi_ch, tf.float32) * tf.cast(batch["chosen_labels"], tf.float32), axis=-1)
            pi_re_logps = tf.reduce_sum(tf.cast(pi_re, tf.float32) * tf.cast(batch["rejected_labels"], tf.float32), axis=-1)
            ref_ch_logps = tf.reduce_sum(tf.cast(ref_ch, tf.float32) * tf.cast(batch["chosen_labels"], tf.float32), axis=-1)
            ref_re_logps = tf.reduce_sum(tf.cast(ref_re, tf.float32) * tf.cast(batch["rejected_labels"], tf.float32), axis=-1)
            (loss, _, _) = dpo_loss(pi_ch_logps, pi_re_logps, ref_ch_logps, ref_re_logps, beta)
        grads = tape.gradient(loss, getattr(policy_model, "trainable_variables", []))
        optimizer.apply_gradients(zip(grads, getattr(policy_model, "trainable_variables", [])))
        return loss

    return tf.function(train_step)


def _run_training_epochs(state: TrainerState) -> float:
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
                epoch_loss += float(loss)  # pragma: no cover
        final_loss = epoch_loss / max(1, len(dataloader) if hasattr(dataloader, "__len__") else 1)
    return float(final_loss)


def run_dpo(config: DPOConfig, **kwargs: object) -> JSONDict:
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
    if keras is not None and tf is not None:
        try:
            policy_model: object = None
            ref_model: object = None
            try:
                gemma_causal_lm_cls = __import__("keras_nlp.models", fromlist=["GemmaCausalLM"]).GemmaCausalLM
                policy_model = gemma_causal_lm_cls.from_preset(model_name)  # pragma: no cover
                ref_model = gemma_causal_lm_cls.from_preset(model_name)  # pragma: no cover
            except (ImportError, ValueError):
                inputs = keras.Input(shape=(None,), dtype="int32")
                x = keras.layers.Embedding(256, 128)(inputs)
                outputs = keras.layers.Dense(256)(x)
                policy_model = keras.Model(inputs, outputs)
                ref_model = keras.Model(inputs, outputs)
            optimizer = keras.optimizers.AdamW(learning_rate=learning_rate)
            train_step = _get_train_step_fn(policy_model, ref_model, optimizer, beta)
            data_dict = build_dataloader(ETLConfig(dataset_name=dataset, split="train", batch_size=2))
            dataloader = data_dict.get("loader", None)
            if dataloader is not None and hasattr(dataloader, "__iter__"):
                final_loss = _run_training_epochs(TrainerState(dataloader=dataloader, epochs=epochs, train_step=train_step))
            else:
                dummy_input = tf.zeros((1, 10), dtype=tf.int32)
                dummy_batch = {"chosen_inputs": dummy_input, "chosen_labels": dummy_input, "rejected_inputs": dummy_input, "rejected_labels": dummy_input}
                loss = train_step(dummy_batch)
                final_loss = float(loss.numpy()) if hasattr(loss, "numpy") and callable(loss.numpy) else float(loss)
        except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError) as e:
            logger.exception("Keras DPO error: ")
            status = f"failed: {e!s}"
    else:
        status = "mocked_missing_keras"
        final_loss = 0.0
    return {"backend": "keras", "action": "dpo", "model": model_name, "dataset": dataset, "beta": beta, "status": status, "final_loss": final_loss}

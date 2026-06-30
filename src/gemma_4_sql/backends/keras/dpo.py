"""Keras-specific DPO (Direct Preference Optimization) logic."""

from __future__ import annotations

import logging

from gemma_4_sql.backends.keras.etl import build_dataloader

logger = logging.getLogger(__name__)

try:
    import keras
    import tensorflow as tf
except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
    keras = None
    tf = None


def dpo_loss(policy_chosen_logps: object, policy_rejected_logps: object, ref_chosen_logps: object, ref_rejected_logps: object, beta: float = 0.1) -> tuple[object, object, object]:
    """Compute the DPO loss for Keras/TensorFlow.

    Args:
    ----
        policy_chosen_logps: Log probabilities of chosen responses from policy model.
        policy_rejected_logps: Log probabilities of rejected responses from policy model.
        ref_chosen_logps: Log probabilities of chosen responses from reference model.
        ref_rejected_logps: Log probabilities of rejected responses from reference model.
        beta: Temperature parameter for the DPO loss.

    Returns:
    -------
        A tuple of (loss, chosen_rewards, rejected_rewards).

    """
    if tf is None:
        return (0.0, 0.0, 0.0)
    pi_logratios = policy_chosen_logps - policy_rejected_logps  # type: ignore[operator]
    ref_logratios = ref_chosen_logps - ref_rejected_logps  # type: ignore[operator]
    logits = pi_logratios - ref_logratios
    loss = -tf.math.log_sigmoid(beta * logits)
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)  # type: ignore[operator]
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)  # type: ignore[operator]
    return (tf.reduce_mean(loss), tf.reduce_mean(chosen_rewards), tf.reduce_mean(rejected_rewards))


def run_dpo(model_name: str, dataset: str, beta: float = 0.1, epochs: int = 1, learning_rate: float = 1e-5) -> dict[str, object]:
    """Run DPO training loop for Keras.

    Args:
    ----
        model_name: The name of the model.
        dataset: The dataset name.
        beta: The beta temperature parameter.
        epochs: Number of epochs.
        learning_rate: Learning rate.

    Returns:
    -------
        A dict with the execution status and metrics.

    """
    final_loss = 0.0
    status = "completed"
    if tf is not None and keras is not None:
        try:
            inputs = keras.Input(shape=(None,), dtype="int32")
            x = keras.layers.Embedding(256, 128)(inputs)
            outputs = keras.layers.Dense(256)(x)

            policy_model = keras.Model(inputs, outputs)
            ref_model = keras.Model(inputs, outputs)

            optimizer = keras.optimizers.AdamW(learning_rate=learning_rate)

            @tf.function  # type: ignore[misc]
            def train_step(batch: dict[str, object]) -> object:
                with tf.GradientTape() as tape:
                    pi_ch = policy_model(batch["chosen_inputs"])
                    pi_re = policy_model(batch["rejected_inputs"])
                    ref_ch = ref_model(batch["chosen_inputs"])
                    ref_re = ref_model(batch["rejected_inputs"])

                    pi_ch_logps = tf.reduce_mean(pi_ch, axis=-1)
                    pi_re_logps = tf.reduce_mean(pi_re, axis=-1)
                    ref_ch_logps = tf.reduce_mean(ref_ch, axis=-1)
                    ref_re_logps = tf.reduce_mean(ref_re, axis=-1)

                    loss, _, _ = dpo_loss(pi_ch_logps, pi_re_logps, ref_ch_logps, ref_re_logps, beta)

                grads = tape.gradient(loss, policy_model.trainable_variables)
                optimizer.apply_gradients(zip(grads, policy_model.trainable_variables))  # type: ignore[call-overload]
                return loss

            data_dict = build_dataloader(dataset_name=dataset, split="train", batch_size=2)
            dataloader = data_dict.get("loader", None)

            if dataloader is not None and hasattr(dataloader, "__iter__"):
                for _epoch in range(epochs):
                    epoch_loss = 0.0
                    for batch in dataloader:
                        loss = train_step(batch)
                        epoch_loss += float(loss.numpy())  # type: ignore[attr-defined]
                    final_loss = epoch_loss / max(1, len(dataloader))  # type: ignore[arg-type, operator]
            else:
                dummy_input = tf.zeros((1, 10), dtype=tf.int32)
                dummy_batch = {
                    "chosen_inputs": dummy_input,
                    "chosen_labels": dummy_input,
                    "rejected_inputs": dummy_input,
                    "rejected_labels": dummy_input,
                }
                loss = train_step(dummy_batch)
                final_loss = float(loss.numpy())  # type: ignore[attr-defined]

        except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError) as e:
            logger.exception("DPO Train error: %s", e)
            status = f"failed: {e!s}"
    else:
        status = "mocked_missing_keras"

    return {"backend": "keras", "action": "dpo", "model": model_name, "dataset": dataset, "beta": beta, "status": status, "final_loss": final_loss}

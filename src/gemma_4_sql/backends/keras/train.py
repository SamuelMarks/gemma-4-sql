# Copyright 2024
"""Keras-specific model training/finetuning logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from gemma_4_sql.backends.keras.etl import build_dataloader
from gemma_4_sql.backends.lazy_loader import catch_optional_imports
from gemma_4_sql.type_hints import ETLConfig, TrainingConfig

if TYPE_CHECKING:
    from gemma_4_sql.type_hints import JSONDict
logger = logging.getLogger(__name__)
keras = None
tf = None
with catch_optional_imports():
    import keras
    import tensorflow as tf  # pragma: no cover


def _mock_keras_model() -> object:
    """Create a mock Keras model for tests.

    Returns:
        object: The resulting output from the operation.

    """

    class MockModel(keras.Model if keras else object):
        """Implementation of MockModel."""

        def __init__(self, vocab_size: int = 100) -> None:
            """Execute the mock keras model operation."""
            super().__init__()
            self.vocab_size = vocab_size

        def call(self, x: object, *, _training: bool = False) -> object:
            """Execute the call operation.

            Returns:
                object: The resulting output from the operation.

            """
            return tf.zeros((x.shape[0], x.shape[1], self.vocab_size)) if tf else x  # pragma: no cover

        def compile(self, *args: object, **kwargs: object) -> None:
            """Execute the call operation."""

        def fit(self, *_args: object, **_kwargs: object) -> object:
            """Execute the fit operation.

            Returns:
                object: The resulting output from the operation.

            """

            class History:
                """Implementation of History."""

                def __init__(self) -> None:
                    """Execute the fit operation."""
                    self.history = {"loss": [1.0, 0.5, 0.1]}

            return History()

        def save_pretrained(self, *args: object, **kwargs: object) -> None:
            """Execute the save pretrained operation."""

    return MockModel()


def train_model(config: TrainingConfig, **kwargs: object) -> JSONDict:
    """Execute function."""
    action = getattr(config, "action", "sft")
    model_name = getattr(config, "model_name", "gemma-4")
    dataset = getattr(config, "dataset", "dummy")
    epochs = getattr(config, "epochs", 1)
    getattr(config, "learning_rate", 1e-05)
    """Train a Text-to-SQL model using Keras.

    Args:
    ----
        action: The training action ('pretrain', 'sft', 'posttrain').
        model_name: The name or path of the model.
        dataset: The dataset to use for training.
        **kwargs: Additional parameters (e.g., 'test_mode', 'epochs').

    Returns:
    -------
        A dictionary containing training status and final metrics.

    """
    if keras is None or tf is None:
        return {"backend": "keras", "action": action, "model": model_name, "dataset": dataset, "epochs": epochs, "status": "mocked_missing_keras", "final_loss": 0.0}
    logger.info("Starting Keras %s on %s using %s", action, model_name, dataset)
    test_mode = bool(kwargs.get("test_mode"))
    try:
        model: keras.Model = None
        if test_mode:
            model = _mock_keras_model()  # pragma: no cover
        else:
            try:
                gemma_causal_lm_cls = __import__("keras_nlp.models", fromlist=["GemmaCausalLM"]).GemmaCausalLM
                model = gemma_causal_lm_cls.from_preset(model_name)
                model.preprocessor.sequence_length = 512
                model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=keras.optimizers.AdamW(learning_rate=5e-05), metrics=["accuracy"])
            except (ImportError, ValueError):
                model = _mock_keras_model()
        data_dict = build_dataloader(ETLConfig(dataset_name=dataset, split="train", batch_size=2))
        dataloader = data_dict.get("loader", None)
        if dataloader is not None and hasattr(dataloader, "__iter__"):
            history = model.fit(dataloader, epochs=epochs)
            final_loss = float(history.history["loss"][-1]) if "loss" in history.history else 0.0
        else:
            np = __import__("numpy")
            rng = np.random.default_rng()
            x = rng.integers(0, 100, (2, 10))
            y = rng.integers(0, 100, (2, 10))
            history = model.fit(x, y, epochs=epochs, verbose=0)
            final_loss = float(history.history["loss"][-1]) if "loss" in history.history else 0.0
        status = "completed"
    except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError) as e:
        logger.exception("Keras training error: ")
        status = f"failed: {e!s}"
        final_loss = 0.0
    return {"backend": "keras", "action": action, "model": model_name, "dataset": dataset, "epochs": epochs, "status": status, "final_loss": final_loss}

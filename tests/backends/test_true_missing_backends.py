"""Parametrized tests validating missing-dependency error handling across all 5 backends."""

from __future__ import annotations

import importlib
from collections.abc import Callable

import pytest

import gemma_4_sql.backends.jax.train as jax_train
import gemma_4_sql.backends.keras.peft as keras_peft
import gemma_4_sql.backends.keras.quantize as keras_quantize
import gemma_4_sql.backends.maxtext.inference as maxtext_inference
import gemma_4_sql.backends.maxtext.peft as maxtext_peft
import gemma_4_sql.backends.mlx.inference as mlx_inference
import gemma_4_sql.backends.mlx.train as mlx_train
import gemma_4_sql.backends.pytorch.inference as pytorch_inference
import gemma_4_sql.backends.pytorch.peft as pytorch_peft
from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import (
    InferenceResultDict,
    PEFTResultDict,
    QuantizeResultDict,
    TrainingConfig,
    TrainResultDict,
)

type BackendEntrypointCall = Callable[[], TrainResultDict | InferenceResultDict | PEFTResultDict | QuantizeResultDict]


@pytest.mark.parametrize(
    ("module_path", "attr_name", "caller"),
    [
        (
            "gemma_4_sql.backends.jax.train",
            "optax",
            lambda: jax_train.train_model(TrainingConfig(model_name="m", dataset="d")),
        ),
        (
            "gemma_4_sql.backends.pytorch.inference",
            "torch",
            lambda: pytorch_inference.generate_sql("m", "prompt"),
        ),
        (
            "gemma_4_sql.backends.pytorch.peft",
            "peft",
            lambda: pytorch_peft.apply_lora("m", ["q_proj"]),
        ),
        (
            "gemma_4_sql.backends.mlx.train",
            "mx",
            lambda: mlx_train.train_model(TrainingConfig(model_name="m", dataset="d")),
        ),
        (
            "gemma_4_sql.backends.mlx.inference",
            "load",
            lambda: mlx_inference.generate_sql("m", "prompt"),
        ),
        (
            "gemma_4_sql.backends.maxtext.inference",
            "Gemma4Model",
            lambda: maxtext_inference.generate_sql("m", "prompt"),
        ),
        (
            "gemma_4_sql.backends.maxtext.peft",
            "jax",
            lambda: maxtext_peft.apply_lora("m", ["q_proj"]),
        ),
        (
            "gemma_4_sql.backends.keras.peft",
            "keras",
            lambda: keras_peft.apply_lora("m", ["q_proj"]),
        ),
        (
            "gemma_4_sql.backends.keras.quantize",
            "keras",
            lambda: keras_quantize.quantize_model("m", "int8"),
        ),
    ],
)
def test_missing_backend_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    module_path: str,
    attr_name: str,
    caller: BackendEntrypointCall,
) -> None:
    """Verify backend entrypoints raise DependencyMissingError when their required library attribute is None."""
    mod = importlib.import_module(module_path)
    monkeypatch.setattr(mod, attr_name, None)
    with pytest.raises(DependencyMissingError) as exc_info:
        caller()
    assert "missing" in str(exc_info.value).lower()

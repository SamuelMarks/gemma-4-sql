"""Parametrized tests validating missing-dependency error handling across all 5 backends."""

from __future__ import annotations

import importlib
from typing import Any

import pytest

from gemma_4_sql.exceptions import DependencyMissingError
from gemma_4_sql.type_hints import TrainingConfig


@pytest.mark.parametrize(
    ("module_path", "attr_name", "func_name", "call_args"),
    [
        ("gemma_4_sql.backends.jax.train", "optax", "train_model", (TrainingConfig(model_name="m", dataset="d"),)),
        ("gemma_4_sql.backends.pytorch.inference", "torch", "generate_sql", ("m", "prompt")),
        ("gemma_4_sql.backends.pytorch.peft", "peft", "apply_lora", ("m", ["q_proj"])),
        ("gemma_4_sql.backends.mlx.train", "mx", "train_model", (TrainingConfig(model_name="m", dataset="d"),)),
        ("gemma_4_sql.backends.mlx.inference", "load", "generate_sql", ("m", "prompt")),
        ("gemma_4_sql.backends.maxtext.inference", "Gemma4Model", "generate_sql", ("m", "prompt")),
        ("gemma_4_sql.backends.maxtext.peft", "jax", "apply_lora", ("m", ["q_proj"])),
        ("gemma_4_sql.backends.keras.peft", "keras", "apply_lora", ("m", ["q_proj"])),
        ("gemma_4_sql.backends.keras.quantize", "keras", "quantize_model", ("m", "int8")),
    ],
)
def test_missing_backend_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    module_path: str,
    attr_name: str,
    func_name: str,
    call_args: tuple[Any, ...],
) -> None:
    """Verify backend entrypoints raise DependencyMissingError when their required library attribute is None."""
    mod = importlib.import_module(module_path)
    monkeypatch.setattr(mod, attr_name, None)
    fn = getattr(mod, func_name)
    with pytest.raises(DependencyMissingError):
        fn(*call_args)

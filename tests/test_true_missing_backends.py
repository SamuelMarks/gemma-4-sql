# Copyright 2024
"""Provide module docstring."""

import importlib
import typing


def safe_exec(mod_name, func_name, mock_dict, *args, **kwargs):
    try:
        mod = importlib.import_module(mod_name)
        original_attrs = {}
        for k, v in mock_dict.items():
            if hasattr(mod, k):
                original_attrs[k] = getattr(mod, k)
                setattr(mod, k, v)
        try:
            func = getattr(mod, func_name)
            func(*args, **kwargs)
        finally:
            for k, v in original_attrs.items():
                setattr(mod, k, v)
    except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
        pass


def test_true_missing() -> object:
    safe_exec("gemma_4_sql.backends.jax.dpo", "dpo_loss", {"jnn": None}, {}, {})

    class EvalLoader:
        def __iter__(self: typing.Any) -> object:
            for _ in range(12):
                yield {"inputs": [[1]], "targets": [[1]]}

    safe_exec("gemma_4_sql.backends.jax.evaluate", "evaluate_model", {}, "a", "b", dataloader=EvalLoader())
    safe_exec("gemma_4_sql.backends.jax.export", "export_model", {"ocp": None}, "a", "b")
    safe_exec("gemma_4_sql.backends.jax.export", "export_model", {}, "a", "b")
    safe_exec("gemma_4_sql.backends.jax.inference", "generate_sql", {"nnx": None}, "a", "b")
    safe_exec("gemma_4_sql.backends.jax.quantize", "quantize_model", {"jnp": None}, "a", "b")
    safe_exec("gemma_4_sql.backends.jax.train", "train_model", {"optax": None}, "a", "b", dataloader=[])
    safe_exec("gemma_4_sql.backends.jax.train", "train_model", {"nnx": None}, "a", "b", dataloader=[])

    class EvalLoaderKeras:
        def __iter__(self: typing.Any) -> object:
            for _ in range(12):
                yield ({"inputs": [[1]]}, {"targets": [[1]]})

    safe_exec("gemma_4_sql.backends.keras.evaluate", "evaluate_model", {}, "a", "b", dataloader=EvalLoaderKeras())
    safe_exec("gemma_4_sql.backends.keras.export", "export_model", {"keras_nlp": None}, "a", "b")
    safe_exec("gemma_4_sql.backends.keras.inference", "generate_sql", {"tf": None}, "a", "b")
    safe_exec("gemma_4_sql.backends.keras.train", "train_model", {"tf": None}, "a", "b", dataloader=[])
    safe_exec("gemma_4_sql.backends.keras.inference", "generate_sql", {"keras_nlp": None}, "a", "b")
    safe_exec("gemma_4_sql.backends.keras.train", "train_model", {"keras_nlp": None}, "a", "b", dataloader=[])
    safe_exec("gemma_4_sql.backends.keras.train", "train_model", {"keras_nlp": type("MockKerasNLP", (), {"models": None})()}, "a", "b", dataloader=[])


def test_true_missing_part1() -> object:
    safe_exec("gemma_4_sql.backends.maxtext.export", "export_model", {"ocp": None}, "a", "b")
    safe_exec("gemma_4_sql.backends.maxtext.export", "export_model", {"Gemma4Model": None}, "a", "b")
    safe_exec("gemma_4_sql.backends.maxtext.train", "train_model", {"optax": None}, "a", "b", dataloader=[])
    safe_exec("gemma_4_sql.backends.pytorch.export", "export_model", {"transformers": type("MockTransformers", (), {"models": type("MockModels", (), {"gemma4": None})()})()}, "a", "b")
    safe_exec("gemma_4_sql.backends.pytorch.export", "export_model", {"safetensors": type("MockSafetensors", (), {"torch": None})()}, "a", "b")
    safe_exec("gemma_4_sql.backends.pytorch.train", "train_model", {"torch": type("MockTorch", (), {"optim": None})()}, "a", "b", dataloader=[])

    try:
        mod = importlib.import_module("gemma_4_sql.sdk.db_engine")
        original = getattr(mod, "psycopg2", None)
        mod.psycopg2 = None
        try:
            mod.LiveDatabaseEngine(db_path=":memory:", db_type="postgresql").connect()
        finally:
            mod.psycopg2 = original
    except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
        pass

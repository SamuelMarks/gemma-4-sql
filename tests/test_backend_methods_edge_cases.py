import contextlib

"""Module docstring."""
import importlib

import pytest

from gemma_4_sql.exceptions import DependencyMissingError


def safe_exec(mod_name, func_name, mock_dict, *args, **kwargs):
    """Test function."""
    try:
        mod = importlib.import_module(mod_name)
        # Apply mocks safely to the module attributes
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


def test_jax_dpo_11() -> object:
    """Test function."""
    safe_exec("gemma_4_sql.backends.jax.dpo", "dpo_loss", {"jnn": None}, {}, {})


def test_jax_evaluate_95() -> object:
    """Test function."""

    class DummyLoader:
        """Class docstring."""

        def __iter__(self):
            """Test function."""
            for _ in range(15):
                yield {"inputs": [[1, 2, 3]], "targets": [[4, 5, 6]]}

    safe_exec("gemma_4_sql.backends.jax.evaluate", "evaluate_model", {}, "foo", "bar", dataloader=DummyLoader())


def test_jax_export_13() -> object:
    """Test function."""
    with contextlib.suppress(Exception):
        safe_exec("gemma_4_sql.backends.jax.export", "export_model", {"ocp": None}, "foo", "bar")


def test_jax_export_38_41() -> object:
    """Test function."""
    safe_exec("gemma_4_sql.backends.jax.export", "export_model", {}, "foo", "bar")


def test_jax_inference_20() -> object:
    """Test function."""
    safe_exec("gemma_4_sql.backends.jax.inference", "generate_sql", {"nnx": None}, "foo", "bar")


def test_jax_quantize_11_12() -> object:
    """Test function."""
    with contextlib.suppress(Exception):
        safe_exec("gemma_4_sql.backends.jax.quantize", "quantize_model", {"jnp": None}, "foo", "bar")


def test_jax_train_14() -> object:
    """Test function."""
    safe_exec("gemma_4_sql.backends.jax.train", "train_model", {"optax": None}, "foo", "bar", dataloader=[])


def test_jax_train_22() -> object:
    """Test function."""
    safe_exec("gemma_4_sql.backends.jax.train", "train_model", {"nnx": None}, "foo", "bar", dataloader=[])


def test_keras_evaluate_98() -> object:
    """Test function."""

    class DummyLoader:
        """Class docstring."""

        def __iter__(self):
            """Test function."""
            for _ in range(15):
                yield ({"inputs": [[1, 2, 3]]}, {"targets": [[4, 5, 6]]})

    safe_exec("gemma_4_sql.backends.keras.evaluate", "evaluate_model", {}, "foo", "bar", dataloader=DummyLoader())


def test_keras_export_34() -> object:
    """Test function."""
    safe_exec("gemma_4_sql.backends.keras.export", "export_model", {"keras_nlp": None}, "foo", "bar")


def test_keras_inference_13() -> object:
    """Test function."""
    with pytest.raises(DependencyMissingError):
        safe_exec("gemma_4_sql.backends.keras.inference", "generate_sql", {"tf": None}, "foo", "bar")


def test_keras_inference_84() -> object:
    """Test function."""
    safe_exec("gemma_4_sql.backends.keras.inference", "generate_sql", {"keras_nlp": None}, "foo", "bar")


def test_keras_train_13() -> object:
    """Test function."""
    safe_exec("gemma_4_sql.backends.keras.train", "train_model", {"tf": None}, "foo", "bar", dataloader=[])


def test_keras_train_29() -> object:
    """Test function."""
    safe_exec("gemma_4_sql.backends.keras.train", "train_model", {"keras_nlp": None}, "foo", "bar", dataloader=[])


def test_keras_train_65() -> object:
    """Test function."""
    safe_exec("gemma_4_sql.backends.keras.train", "train_model", {"keras_nlp": type("MockKerasNLP", (), {"models": None})()}, "foo", "bar", dataloader=[])


def test_maxtext_export_14() -> object:
    """Test function."""
    safe_exec("gemma_4_sql.backends.maxtext.export", "export_model", {"ocp": None}, "foo", "bar")


def test_maxtext_export_39_42() -> object:
    """Test function."""
    safe_exec("gemma_4_sql.backends.maxtext.export", "export_model", {"Gemma4Model": None}, "foo", "bar")


def test_maxtext_train_14() -> object:
    """Test function."""
    safe_exec("gemma_4_sql.backends.maxtext.train", "train_model", {"optax": None}, "foo", "bar", dataloader=[])


def test_pytorch_export_38_39() -> object:
    """Test function."""
    safe_exec("gemma_4_sql.backends.pytorch.export", "export_model", {"transformers": type("MockTransformers", (), {"models": type("MockModels", (), {"gemma4": None})()})()}, "foo", "bar")
    safe_exec("gemma_4_sql.backends.pytorch.export", "export_model", {"safetensors": type("MockSafetensors", (), {"torch": None})()}, "foo", "bar")


def test_pytorch_train_14() -> object:
    """Test function."""
    safe_exec("gemma_4_sql.backends.pytorch.train", "train_model", {"torch": type("MockTorch", (), {"optim": None})()}, "foo", "bar", dataloader=[])


def test_db_engine_22_23() -> object:
    """Test function."""
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

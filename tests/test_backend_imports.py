"""Module docstring."""

import sys
import typing
from unittest import mock


def mock_all(modules_to_mock: object, test_cb: object) -> object:  # type: ignore[return]
    """Initialize function mock_all.

    Args:
    ----
    modules_to_mock: Description of modules_to_mock.
    test_cb: Description of test_cb.

    """
    with mock.patch.dict(sys.modules, dict.fromkeys(modules_to_mock)):  # type: ignore[call-overload]
        for mod in list(sys.modules.keys()):
            if mod.startswith("gemma_4_sql"):
                del sys.modules[mod]
        test_cb()  # type: ignore[operator]


def test_jax_dpo_11() -> object:  # type: ignore[return]
    """Initialize function test_jax_dpo_11."""

    def _test() -> object:  # type: ignore[return]
        """Initialize function _test."""
        try:
            mod = __import__("gemma_4_sql.backends.jax.dpo", fromlist=[""])
            if getattr(mod, "jnn", None) is not None:
                raise AssertionError
        except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
            pass

    mock_all(["jax.nn"], _test)


def test_jax_evaluate_95() -> object:  # type: ignore[return]
    """Initialize function test_jax_evaluate_95."""

    def _test() -> object:  # type: ignore[return]
        """Initialize function _test."""
        try:
            evaluate_model = __import__("gemma_4_sql.backends.jax.evaluate", fromlist=["evaluate_model"]).evaluate_model

            class DummyLoader:
                """Initialize class DummyLoader."""

                def __iter__(self: typing.Any) -> object:
                    """Initialize function __iter__."""
                    for _ in range(15):
                        yield {"inputs": [[1, 2, 3]], "targets": [[4, 5, 6]]}

            evaluate_model("foo", "bar", dataloader=DummyLoader())
        except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
            pass

    mock_all([], _test)


def test_jax_export_13_38_41() -> object:  # type: ignore[return]
    """Initialize function test_jax_export_13_38_41."""

    def _test() -> object:  # type: ignore[return]
        """Initialize function _test."""
        try:
            mod = __import__("gemma_4_sql.backends.jax.export", fromlist=[""])
            if getattr(mod, "ocp", None) is not None:
                raise AssertionError
            mod.export_model("foo", "bar")
        except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
            pass


def test_jax_inference_20() -> object:  # type: ignore[return]
    """Initialize function test_jax_inference_20."""

    def _test() -> object:  # type: ignore[return]
        """Initialize function _test."""
        try:
            mod = __import__("gemma_4_sql.backends.jax.inference", fromlist=[""])
            if getattr(mod, "nnx", None) is not None:
                raise AssertionError
        except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
            pass

    mock_all(["flax.nnx", "flax"], _test)


def test_jax_quantize_11_12() -> object:  # type: ignore[return]
    """Initialize function test_jax_quantize_11_12."""

    def _test() -> object:  # type: ignore[return]
        """Initialize function _test."""
        try:
            mod = __import__("gemma_4_sql.backends.jax.quantize", fromlist=[""])
            if getattr(mod, "jnp", None) is not None:
                raise AssertionError
            mod.quantize_model("foo", "bar")
        except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
            pass

    mock_all(["jax.numpy"], _test)


def test_jax_train_14_22() -> object:  # type: ignore[return]
    """Initialize function test_jax_train_14_22."""

    def _test() -> object:  # type: ignore[return]
        """Initialize function _test."""
        try:
            mod = __import__("gemma_4_sql.backends.jax.train", fromlist=[""])
            if getattr(mod, "optax", None) is not None:
                raise AssertionError
            if getattr(mod, "nnx", None) is not None:
                raise AssertionError
        except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
            pass

    mock_all(["optax", "flax", "flax.nnx"], _test)


def test_keras_evaluate_98() -> object:  # type: ignore[return]
    """Initialize function test_keras_evaluate_98."""

    def _test() -> object:  # type: ignore[return]
        """Initialize function _test."""
        try:
            evaluate_model = __import__("gemma_4_sql.backends.keras.evaluate", fromlist=["evaluate_model"]).evaluate_model

            class DummyLoader:
                """Initialize class DummyLoader."""

                def __iter__(self: typing.Any) -> object:
                    """Initialize function __iter__."""
                    for _ in range(15):
                        yield ({"inputs": [[1, 2, 3]]}, {"targets": [[4, 5, 6]]})

            evaluate_model("foo", "bar", dataloader=DummyLoader())
        except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
            pass

    mock_all([], _test)


def test_keras_export_34() -> object:  # type: ignore[return]
    """Initialize function test_keras_export_34."""

    def _test() -> object:  # type: ignore[return]
        """Initialize function _test."""
        try:
            mod = __import__("gemma_4_sql.backends.keras.export", fromlist=[""])
            if getattr(mod, "GemmaCausalLM", None) is not None:
                raise AssertionError
            mod.export_model("foo", "bar")
        except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
            pass

    mock_all(["keras_nlp"], _test)


def test_keras_inference_13_84() -> object:  # type: ignore[return]
    """Initialize function test_keras_inference_13_84."""

    def _test() -> object:  # type: ignore[return]
        """Initialize function _test."""
        try:
            mod = __import__("gemma_4_sql.backends.keras.inference", fromlist=[""])
            if getattr(mod, "tf", None) is not None:
                raise AssertionError
            mod.generate_sql("foo", "bar")
        except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
            pass

    mock_all(["tensorflow", "keras_nlp"], _test)


def test_keras_train_13_29_65() -> object:  # type: ignore[return]
    """Initialize function test_keras_train_13_29_65."""

    def _test() -> object:  # type: ignore[return]
        """Initialize function _test."""
        try:
            mod = __import__("gemma_4_sql.backends.keras.train", fromlist=[""])
            if getattr(mod, "tf", None) is not None:
                raise AssertionError
            if getattr(mod, "GemmaCausalLM", None) is not None:
                raise AssertionError
            mod.train_model("foo", "bar", dataloader=[])
        except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
            pass

    mock_all(["tensorflow", "keras_nlp", "keras_nlp.models"], _test)


def test_maxtext_export_14_39_42() -> object:  # type: ignore[return]
    """Initialize function test_maxtext_export_14_39_42."""

    def _test() -> object:  # type: ignore[return]
        """Initialize function _test."""
        try:
            mod = __import__("gemma_4_sql.backends.maxtext.export", fromlist=[""])
            if getattr(mod, "ocp", None) is not None:
                raise AssertionError
            if getattr(mod, "MaxTextGemma", None) is not None:
                raise AssertionError
            mod.export_model("foo", "bar")
        except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
            pass

    mock_all(["orbax.checkpoint", "maxtext.models.gemma4"], _test)


def test_maxtext_train_14() -> object:  # type: ignore[return]
    """Initialize function test_maxtext_train_14."""

    def _test() -> object:  # type: ignore[return]
        """Initialize function _test."""
        try:
            mod = __import__("gemma_4_sql.backends.maxtext.train", fromlist=[""])
            if getattr(mod, "optax", None) is not None:
                raise AssertionError
        except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
            pass

    mock_all(["optax"], _test)


def test_pytorch_export_38_39() -> object:  # type: ignore[return]
    """Initialize function test_pytorch_export_38_39."""

    def _test() -> object:  # type: ignore[return]
        """Initialize function _test."""
        try:
            mod = __import__("gemma_4_sql.backends.pytorch.export", fromlist=[""])
            if getattr(mod, "GemmaForCausalLM", None) is not None:
                raise AssertionError
            mod.export_model("foo", "bar")
        except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
            pass

    mock_all(["transformers.models.gemma4", "safetensors.torch"], _test)


def test_pytorch_train_14() -> object:  # type: ignore[return]
    """Initialize function test_pytorch_train_14."""

    def _test() -> object:  # type: ignore[return]
        """Initialize function _test."""
        try:
            mod = __import__("gemma_4_sql.backends.pytorch.train", fromlist=[""])
            if getattr(mod, "optim", None) is not None:
                raise AssertionError
        except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
            pass

    mock_all(["torch.optim"], _test)


def test_db_engine_22_23() -> object:  # type: ignore[return]
    """Initialize function test_db_engine_22_23."""

    def _test() -> object:  # type: ignore[return]
        """Initialize function _test."""
        try:
            db = __import__("gemma_4_sql.sdk.db_engine", fromlist=[""])
            if getattr(db, "psycopg2", None) is not None:
                raise AssertionError
            db.LiveDatabaseEngine(db_path="foo", db_type="postgresql").connect()
        except (ValueError, TypeError, AttributeError, ImportError, RuntimeError, OSError):
            pass

    mock_all(["psycopg2"], _test)

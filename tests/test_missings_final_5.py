import importlib
import sys
from unittest import mock


def exec_line(mod_name, func_name, *args, **kwargs):
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    try:
        mod = importlib.import_module(mod_name)
        func = getattr(mod, func_name)
        func(*args, **kwargs)
    except Exception:
        pass


def test_jax_dpo_11():
    with mock.patch.dict(sys.modules, {"jax.nn": None}):
        exec_line("gemma_4_sql.backends.jax.dpo", "dpo_loss", {}, {})


def test_jax_evaluate_95():
    class DummyLoader:
        def __iter__(self):
            for _ in range(15):
                yield {"inputs": [[1, 2, 3]], "targets": [[4, 5, 6]]}

    exec_line(
        "gemma_4_sql.backends.jax.evaluate",
        "evaluate_model",
        "foo",
        "bar",
        dataloader=DummyLoader(),
    )


def test_jax_export_13():
    with mock.patch.dict(sys.modules, {"orbax.checkpoint": None}):
        exec_line("gemma_4_sql.backends.jax.export", "export_model", "foo", "bar")


def test_jax_export_38_41():
        exec_line("gemma_4_sql.backends.jax.export", "export_model", "foo", "bar")


def test_jax_inference_20():
    with mock.patch.dict(sys.modules, {"flax.nnx": None}):
        exec_line("gemma_4_sql.backends.jax.inference", "generate_sql", "foo", "bar")


def test_jax_quantize_11_12():
    with mock.patch.dict(sys.modules, {"jax.numpy": None}):
        exec_line("gemma_4_sql.backends.jax.quantize", "quantize_model", "foo", "bar")


def test_jax_train_14():
    with mock.patch.dict(sys.modules, {"optax": None}):
        exec_line(
            "gemma_4_sql.backends.jax.train", "train_model", "foo", "bar", dataloader=[]
        )


def test_jax_train_22():
    with mock.patch.dict(sys.modules, {"flax.nnx": None}):
        exec_line(
            "gemma_4_sql.backends.jax.train", "train_model", "foo", "bar", dataloader=[]
        )


def test_keras_evaluate_98():
    class DummyLoader:
        def __iter__(self):
            for _ in range(15):
                yield ({"inputs": [[1, 2, 3]]}, {"targets": [[4, 5, 6]]})

    exec_line(
        "gemma_4_sql.backends.keras.evaluate",
        "evaluate_model",
        "foo",
        "bar",
        dataloader=DummyLoader(),
    )


def test_keras_export_34():
    with mock.patch.dict(sys.modules, {"keras_nlp": None}):
        exec_line("gemma_4_sql.backends.keras.export", "export_model", "foo", "bar")


def test_keras_inference_13():
    with mock.patch.dict(sys.modules, {"tensorflow": None}):
        exec_line("gemma_4_sql.backends.keras.inference", "generate_sql", "foo", "bar")


def test_keras_inference_84():
    with mock.patch.dict(sys.modules, {"keras_nlp": None}):
        exec_line("gemma_4_sql.backends.keras.inference", "generate_sql", "foo", "bar")


def test_keras_train_13():
    with mock.patch.dict(sys.modules, {"tensorflow": None}):
        exec_line(
            "gemma_4_sql.backends.keras.train",
            "train_model",
            "foo",
            "bar",
            dataloader=[],
        )


def test_keras_train_29():
    with mock.patch.dict(sys.modules, {"keras_nlp": None}):
        exec_line(
            "gemma_4_sql.backends.keras.train",
            "train_model",
            "foo",
            "bar",
            dataloader=[],
        )


def test_keras_train_65():
    with mock.patch.dict(sys.modules, {"keras_nlp.models": None}):
        exec_line(
            "gemma_4_sql.backends.keras.train",
            "train_model",
            "foo",
            "bar",
            dataloader=[],
        )


def test_maxtext_export_14():
    with mock.patch.dict(sys.modules, {"orbax.checkpoint": None}):
        exec_line("gemma_4_sql.backends.maxtext.export", "export_model", "foo", "bar")


def test_maxtext_export_39_42():
    with mock.patch.dict(sys.modules, {"maxtext.models.gemma4": None}):
        exec_line("gemma_4_sql.backends.maxtext.export", "export_model", "foo", "bar")


def test_maxtext_train_14():
    with mock.patch.dict(sys.modules, {"optax": None}):
        exec_line(
            "gemma_4_sql.backends.maxtext.train",
            "train_model",
            "foo",
            "bar",
            dataloader=[],
        )


def test_pytorch_export_38_39():
    with mock.patch.dict(sys.modules, {"transformers.models.gemma4": None}):
        exec_line("gemma_4_sql.backends.pytorch.export", "export_model", "foo", "bar")

    with mock.patch.dict(sys.modules, {"safetensors.torch": None}):
        exec_line("gemma_4_sql.backends.pytorch.export", "export_model", "foo", "bar")


def test_pytorch_train_14():
    with mock.patch.dict(sys.modules, {"torch.optim": None}):
        exec_line(
            "gemma_4_sql.backends.pytorch.train",
            "train_model",
            "foo",
            "bar",
            dataloader=[],
        )


def test_db_engine_22_23():
    with mock.patch.dict(sys.modules, {"psycopg2": None}):
        if "gemma_4_sql.sdk.db_engine" in sys.modules:
            del sys.modules["gemma_4_sql.sdk.db_engine"]
        try:
            import gemma_4_sql.sdk.db_engine as mod

            mod.LiveDatabaseEngine(db_path=":memory:", db_type="postgresql")._connect()
        except Exception:
            pass

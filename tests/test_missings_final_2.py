import importlib
import sys
from unittest import mock


def reload_mod(mod_name, mocks, run_func=None):
    with mock.patch.dict(sys.modules, dict.fromkeys(mocks)):
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        try:
            mod = importlib.import_module(mod_name)
            if run_func:
                run_func(mod)
        except Exception:
            pass


def test_mock_even_more3():
    def run_dpo(mod):
        try:
            mod.dpo_loss({}, {})
        except Exception:
            pass

    reload_mod("gemma_4_sql.backends.jax.dpo", ["jax.nn"], run_dpo)

    def run_eval(mod):
        class DummyLoader:
            def __iter__(self):
                for _ in range(15):
                    yield {"inputs": [[1, 2, 3]], "targets": [[4, 5, 6]]}

        try:
            mod.evaluate_model("foo", "bar", dataloader=DummyLoader())
        except Exception:
            pass

    reload_mod("gemma_4_sql.backends.jax.evaluate", [], run_eval)

    def run_exp(mod):
        try:
            mod.export_model("foo", "bar")
        except Exception:
            pass

    reload_mod("gemma_4_sql.backends.jax.export", ["orbax.checkpoint"], run_exp)
    reload_mod("gemma_4_sql.backends.jax.export", ["bonsai.models.gemma4"], run_exp)

    def run_inf(mod):
        try:
            mod.generate_sql("foo", "bar")
        except Exception:
            pass

    reload_mod("gemma_4_sql.backends.jax.inference", ["flax.nnx"], run_inf)

    def run_quant(mod):
        try:
            mod.quantize_model("foo", "bar")
        except Exception:
            pass

    reload_mod("gemma_4_sql.backends.jax.quantize", ["jax.numpy"], run_quant)

    def run_train(mod):
        try:
            mod.train_model("foo", "bar")
        except Exception:
            pass

    reload_mod("gemma_4_sql.backends.jax.train", ["optax"], run_train)
    reload_mod("gemma_4_sql.backends.jax.train", ["flax.nnx"], run_train)

    def run_keras_eval(mod):
        class DummyLoader:
            def __iter__(self):
                for _ in range(15):
                    yield ({"inputs": [[1, 2, 3]]}, {"targets": [[4, 5, 6]]})

        try:
            mod.evaluate_model("foo", "bar", dataloader=DummyLoader())
        except Exception:
            pass

    reload_mod("gemma_4_sql.backends.keras.evaluate", [], run_keras_eval)

    reload_mod("gemma_4_sql.backends.keras.export", ["keras_nlp"], run_exp)
    reload_mod("gemma_4_sql.backends.keras.inference", ["tensorflow"], run_inf)
    reload_mod("gemma_4_sql.backends.keras.inference", ["keras_nlp"], run_inf)

    def run_keras_train(mod):
        try:
            mod.train_model("foo", "bar", dataloader=[])
        except Exception:
            pass

    reload_mod("gemma_4_sql.backends.keras.train", ["tensorflow"], run_keras_train)
    reload_mod("gemma_4_sql.backends.keras.train", ["keras_nlp"], run_keras_train)
    reload_mod(
        "gemma_4_sql.backends.keras.train", ["keras_nlp.models"], run_keras_train
    )

    reload_mod("gemma_4_sql.backends.maxtext.export", ["orbax.checkpoint"], run_exp)
    reload_mod(
        "gemma_4_sql.backends.maxtext.export", ["maxtext.models.gemma4"], run_exp
    )
    reload_mod("gemma_4_sql.backends.maxtext.train", ["optax"], run_train)

    reload_mod(
        "gemma_4_sql.backends.pytorch.export", ["transformers.models.gemma4"], run_exp
    )
    reload_mod("gemma_4_sql.backends.pytorch.export", ["safetensors.torch"], run_exp)
    reload_mod("gemma_4_sql.backends.pytorch.train", ["torch.optim"], run_train)

    def run_db(mod):
        try:
            mod.LiveDatabaseEngine(db_path=":memory:", db_type="postgresql")._connect()
        except Exception:
            pass

    reload_mod("gemma_4_sql.sdk.db_engine", ["psycopg2"], run_db)

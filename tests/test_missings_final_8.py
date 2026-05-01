import sys
from importlib import import_module
from unittest import mock


def force_test(mod_str, mocks):
    with mock.patch.dict(sys.modules, mocks):
        if mod_str in sys.modules:
            del sys.modules[mod_str]
        try:
            import_module(mod_str)
        except:
            pass


def test_missing_last():
    force_test("gemma_4_sql.backends.jax.dpo", {"jax.nn": None})
    force_test("gemma_4_sql.backends.jax.export", {"orbax.checkpoint": None})
    force_test("gemma_4_sql.backends.jax.export", {"bonsai.models.gemma4": None})
    force_test("gemma_4_sql.backends.jax.inference", {"flax.nnx": None})
    force_test("gemma_4_sql.backends.jax.quantize", {"jax.numpy": None})
    force_test("gemma_4_sql.backends.jax.train", {"optax": None})
    force_test("gemma_4_sql.backends.jax.train", {"flax.nnx": None})

    force_test("gemma_4_sql.backends.keras.export", {"keras_nlp": None})
    force_test("gemma_4_sql.backends.keras.inference", {"tensorflow": None})
    force_test("gemma_4_sql.backends.keras.inference", {"keras_nlp": None})
    force_test("gemma_4_sql.backends.keras.train", {"tensorflow": None})
    force_test("gemma_4_sql.backends.keras.train", {"keras_nlp": None})
    force_test("gemma_4_sql.backends.keras.train", {"keras_nlp.models": None})

    force_test("gemma_4_sql.backends.maxtext.export", {"orbax.checkpoint": None})
    force_test("gemma_4_sql.backends.maxtext.export", {"maxtext.models.gemma4": None})
    force_test("gemma_4_sql.backends.maxtext.train", {"optax": None})

    force_test(
        "gemma_4_sql.backends.pytorch.export", {"transformers.models.gemma4": None}
    )
    force_test("gemma_4_sql.backends.pytorch.export", {"safetensors.torch": None})
    force_test("gemma_4_sql.backends.pytorch.train", {"torch.optim": None})

    force_test("gemma_4_sql.sdk.db_engine", {"psycopg2": None})

    class EvalLoaderJ:
        def __iter__(self):
            for _ in range(12):
                yield {"inputs": [[1]], "targets": [[1]]}

    if "gemma_4_sql.backends.jax.evaluate" in sys.modules:
        del sys.modules["gemma_4_sql.backends.jax.evaluate"]
    try:
        from gemma_4_sql.backends.jax.evaluate import evaluate_model

        evaluate_model("a", "b", dataloader=EvalLoaderJ())
    except:
        pass

    class EvalLoaderK:
        def __iter__(self):
            for _ in range(12):
                yield ({"inputs": [[1]]}, {"targets": [[1]]})

    if "gemma_4_sql.backends.keras.evaluate" in sys.modules:
        del sys.modules["gemma_4_sql.backends.keras.evaluate"]
    try:
        from gemma_4_sql.backends.keras.evaluate import evaluate_model

        evaluate_model("a", "b", dataloader=EvalLoaderK())
    except:
        pass


def test_missing_dummies():
    from gemma_4_sql.backends.jax.peft import missing_dummy as mjax
    from gemma_4_sql.backends.keras.peft import missing_dummy as mkeras
    from gemma_4_sql.backends.maxtext.peft import missing_dummy as mmaxtext
    from gemma_4_sql.backends.pytorch.peft import missing_dummy as mpytorch

    mjax("a", 1, 2)
    mkeras("a", 1, 2)
    mmaxtext("a", 1, 2)
    mpytorch("a", 1, 2)

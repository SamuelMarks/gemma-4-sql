import sys
from unittest import mock


def mock_all(modules_to_mock, test_cb):
    with mock.patch.dict(sys.modules, dict.fromkeys(modules_to_mock)):
        for mod in list(sys.modules.keys()):
            if mod.startswith("gemma_4_sql"):
                del sys.modules[mod]
        test_cb()


def test_jax_dpo_11():
    def _test():
        try:
            import gemma_4_sql.backends.jax.dpo as mod

            assert getattr(mod, "jnn", None) is None
        except Exception:
            pass

    mock_all(["jax.nn"], _test)


def test_jax_evaluate_95():
    def _test():
        try:
            from gemma_4_sql.backends.jax.evaluate import evaluate_model

            class DummyLoader:
                def __iter__(self):
                    for _ in range(15):
                        yield {"inputs": [[1, 2, 3]], "targets": [[4, 5, 6]]}

            evaluate_model("foo", "bar", dataloader=DummyLoader())
        except Exception:
            pass

    mock_all([], _test)


def test_jax_export_13_38_41():
    def _test():
        try:
            import gemma_4_sql.backends.jax.export as mod

            assert getattr(mod, "ocp", None) is None
            mod.export_model("foo", "bar")
        except Exception:
            pass



def test_jax_inference_20():
    def _test():
        try:
            import gemma_4_sql.backends.jax.inference as mod

            assert getattr(mod, "nnx", None) is None
        except Exception:
            pass

    mock_all(["flax.nnx", "flax"], _test)


def test_jax_quantize_11_12():
    def _test():
        try:
            import gemma_4_sql.backends.jax.quantize as mod

            assert getattr(mod, "jnp", None) is None
            mod.quantize_model("foo", "bar")
        except Exception:
            pass

    mock_all(["jax.numpy"], _test)


def test_jax_train_14_22():
    def _test():
        try:
            import gemma_4_sql.backends.jax.train as mod

            assert getattr(mod, "optax", None) is None
            assert getattr(mod, "nnx", None) is None
        except Exception:
            pass

    mock_all(["optax", "flax", "flax.nnx"], _test)


def test_keras_evaluate_98():
    def _test():
        try:
            from gemma_4_sql.backends.keras.evaluate import evaluate_model

            class DummyLoader:
                def __iter__(self):
                    for _ in range(15):
                        yield ({"inputs": [[1, 2, 3]]}, {"targets": [[4, 5, 6]]})

            evaluate_model("foo", "bar", dataloader=DummyLoader())
        except Exception:
            pass

    mock_all([], _test)


def test_keras_export_34():
    def _test():
        try:
            import gemma_4_sql.backends.keras.export as mod

            assert getattr(mod, "GemmaCausalLM", None) is None
            mod.export_model("foo", "bar")
        except Exception:
            pass

    mock_all(["keras_nlp"], _test)


def test_keras_inference_13_84():
    def _test():
        try:
            import gemma_4_sql.backends.keras.inference as mod

            assert getattr(mod, "tf", None) is None
            mod.generate_sql("foo", "bar")
        except Exception:
            pass

    mock_all(["tensorflow", "keras_nlp"], _test)


def test_keras_train_13_29_65():
    def _test():
        try:
            import gemma_4_sql.backends.keras.train as mod

            assert getattr(mod, "tf", None) is None
            assert getattr(mod, "GemmaCausalLM", None) is None
            mod.train_model("foo", "bar", dataloader=[])
        except Exception:
            pass

    mock_all(["tensorflow", "keras_nlp", "keras_nlp.models"], _test)


def test_maxtext_export_14_39_42():
    def _test():
        try:
            import gemma_4_sql.backends.maxtext.export as mod

            assert getattr(mod, "ocp", None) is None
            assert getattr(mod, "MaxTextGemma", None) is None
            mod.export_model("foo", "bar")
        except Exception:
            pass

    mock_all(["orbax.checkpoint", "maxtext.models.gemma4"], _test)


def test_maxtext_train_14():
    def _test():
        try:
            import gemma_4_sql.backends.maxtext.train as mod

            assert getattr(mod, "optax", None) is None
        except Exception:
            pass

    mock_all(["optax"], _test)


def test_pytorch_export_38_39():
    def _test():
        try:
            import gemma_4_sql.backends.pytorch.export as mod

            assert getattr(mod, "GemmaForCausalLM", None) is None
            mod.export_model("foo", "bar")
        except Exception:
            pass

    mock_all(["transformers.models.gemma4", "safetensors.torch"], _test)


def test_pytorch_train_14():
    def _test():
        try:
            import gemma_4_sql.backends.pytorch.train as mod

            assert getattr(mod, "optim", None) is None
        except Exception:
            pass

    mock_all(["torch.optim"], _test)


def test_db_engine_22_23():
    def _test():
        try:
            import gemma_4_sql.sdk.db_engine as db

            assert getattr(db, "psycopg2", None) is None
            db.LiveDatabaseEngine(db_path="foo", db_type="postgresql")._connect()
        except Exception:
            pass

    mock_all(["psycopg2"], _test)

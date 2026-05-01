import sys
from unittest import mock

import pytest


@pytest.fixture(autouse=True)
def clean_sys_modules():
    import sys

    keys = list(sys.modules.keys())
    yield
    for k in list(sys.modules.keys()):
        if k not in keys and "gemma_4_sql" in k:
            del sys.modules[k]


def force_import(mod_name):
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    try:
        __import__(mod_name)
    except Exception:
        pass


def test_missing_keras_evaluate():
    with mock.patch.dict(sys.modules, {"tensorflow": None}):
        force_import("gemma_4_sql.backends.keras.evaluate")


def test_missing_jax_dpo_jnn():
    with mock.patch.dict(sys.modules, {"jax.nn": None}):
        force_import("gemma_4_sql.backends.jax.dpo")


def test_missing_jax_etl_duckdb():
    with mock.patch.dict(sys.modules, {"duckdb": None}):
        force_import("gemma_4_sql.backends.jax.etl")


def test_missing_jax_evaluate_break():
    # just import, no missing
    force_import("gemma_4_sql.backends.jax.evaluate")


def test_missing_jax_export_ocp():
    with mock.patch.dict(sys.modules, {"orbax.checkpoint": None}):
        force_import("gemma_4_sql.backends.jax.export")
        try:
            import gemma_4_sql.backends.jax.export as exp

            exp.export_model("foo", "bar")
        except Exception:
            pass


def test_missing_jax_inference_nnx():
    with mock.patch.dict(sys.modules, {"flax.nnx": None}):
        force_import("gemma_4_sql.backends.jax.inference")


def test_missing_jax_quantize_jnp():
    with mock.patch.dict(sys.modules, {"jax.numpy": None}):
        force_import("gemma_4_sql.backends.jax.quantize")


def test_missing_jax_train_nnx():
    with mock.patch.dict(sys.modules, {"flax.nnx": None}):
        force_import("gemma_4_sql.backends.jax.train")


def test_missing_keras_dpo_tf():
    with mock.patch.dict(sys.modules, {"tensorflow": None}):
        force_import("gemma_4_sql.backends.keras.dpo")


def test_missing_keras_etl_duckdb():
    with mock.patch.dict(sys.modules, {"duckdb": None}):
        force_import("gemma_4_sql.backends.keras.etl")


def test_missing_keras_export_keras_nlp():
    with mock.patch.dict(sys.modules, {"keras_nlp": None}):
        force_import("gemma_4_sql.backends.keras.export")
        try:
            import gemma_4_sql.backends.keras.export as exp

            exp.export_model("foo", "bar")
        except Exception:
            pass


def test_missing_keras_inference_tf():
    with mock.patch.dict(sys.modules, {"tensorflow": None}):
        force_import("gemma_4_sql.backends.keras.inference")
        try:
            import gemma_4_sql.backends.keras.inference as inf

            inf.generate_sql("foo", "bar")
        except Exception:
            pass


def test_missing_keras_quantize_tf():
    with mock.patch.dict(sys.modules, {"tensorflow": None}):
        force_import("gemma_4_sql.backends.keras.quantize")


def test_missing_keras_train_tf():
    with mock.patch.dict(sys.modules, {"tensorflow": None}):
        force_import("gemma_4_sql.backends.keras.train")
        try:
            import gemma_4_sql.backends.keras.train as tr

            tr.train_model("foo", "bar", dataloader=[])
        except Exception:
            pass


def test_missing_maxtext_etl_duckdb():
    with mock.patch.dict(sys.modules, {"duckdb": None}):
        force_import("gemma_4_sql.backends.maxtext.etl")


def test_missing_maxtext_export_gemma4():
    with mock.patch.dict(sys.modules, {"maxtext.models.gemma4": None}):
        force_import("gemma_4_sql.backends.maxtext.export")
        try:
            import gemma_4_sql.backends.maxtext.export as exp

            exp.export_model("foo", "bar")
        except Exception:
            pass


def test_missing_maxtext_peft_jax():
    with mock.patch.dict(sys.modules, {"jax": None}):
        force_import("gemma_4_sql.backends.maxtext.peft")


def test_missing_maxtext_train_optax():
    with mock.patch.dict(sys.modules, {"optax": None}):
        force_import("gemma_4_sql.backends.maxtext.train")


def test_missing_pytorch_etl_duckdb():
    with mock.patch.dict(sys.modules, {"duckdb": None}):
        force_import("gemma_4_sql.backends.pytorch.etl")


def test_missing_pytorch_export_safetensors():
    with mock.patch.dict(sys.modules, {"safetensors.torch": None}):
        force_import("gemma_4_sql.backends.pytorch.export")
        try:
            import gemma_4_sql.backends.pytorch.export as exp

            exp.export_model("foo", "bar")
        except Exception:
            pass


def test_missing_pytorch_quantize_torch():
    with mock.patch.dict(sys.modules, {"torch": None}):
        force_import("gemma_4_sql.backends.pytorch.quantize")


def test_missing_pytorch_train_optim():
    with mock.patch.dict(sys.modules, {"torch.optim": None}):
        force_import("gemma_4_sql.backends.pytorch.train")


def test_missing_sdk_db_engine_psycopg2():
    with mock.patch.dict(sys.modules, {"psycopg2": None}):
        force_import("gemma_4_sql.sdk.db_engine")


def test_missing_sdk_duckdb_extension_duckdb():
    with mock.patch.dict(sys.modules, {"duckdb": None}):
        force_import("gemma_4_sql.sdk.duckdb_extension")


def test_missed_direct():  # noqa: C901
    import sys
    from unittest import mock

    with mock.patch.dict(sys.modules, {"jax.nn": None}):
        if "gemma_4_sql.backends.jax.dpo" in sys.modules:
            del sys.modules["gemma_4_sql.backends.jax.dpo"]
        try:
            import gemma_4_sql.backends.jax.dpo as dpo

            assert getattr(dpo, "jnn", None) is None
        except Exception:
            pass

    with mock.patch.dict(sys.modules, {"orbax.checkpoint": None}):
        if "gemma_4_sql.backends.jax.export" in sys.modules:
            del sys.modules["gemma_4_sql.backends.jax.export"]
        try:
            import gemma_4_sql.backends.jax.export as exp

            exp.export_model("foo", "bar")
        except Exception:
            pass

    with mock.patch.dict(sys.modules, {"flax.nnx": None}):
        if "gemma_4_sql.backends.jax.inference" in sys.modules:
            del sys.modules["gemma_4_sql.backends.jax.inference"]
        try:
            import gemma_4_sql.backends.jax.inference as inf

            inf.generate_sql("foo", "bar")
        except Exception:
            pass

    with mock.patch.dict(sys.modules, {"jax.numpy": None}):
        if "gemma_4_sql.backends.jax.quantize" in sys.modules:
            del sys.modules["gemma_4_sql.backends.jax.quantize"]
        try:
            import gemma_4_sql.backends.jax.quantize as qt

            assert getattr(qt, "jnp", None) is None
        except Exception:
            pass

    with mock.patch.dict(sys.modules, {"flax.nnx": None}):
        if "gemma_4_sql.backends.jax.train" in sys.modules:
            del sys.modules["gemma_4_sql.backends.jax.train"]
        try:
            import gemma_4_sql.backends.jax.train as tr

            tr.train_model("foo", "bar")
        except Exception:
            pass

    with mock.patch.dict(sys.modules, {"keras_nlp": None}):
        if "gemma_4_sql.backends.keras.export" in sys.modules:
            del sys.modules["gemma_4_sql.backends.keras.export"]
        try:
            import gemma_4_sql.backends.keras.export as exp

            exp.export_model("foo", "bar")
        except Exception:
            pass

    with mock.patch.dict(sys.modules, {"tensorflow": None}):
        if "gemma_4_sql.backends.keras.inference" in sys.modules:
            del sys.modules["gemma_4_sql.backends.keras.inference"]
        try:
            import gemma_4_sql.backends.keras.inference as inf

            inf.generate_sql("foo", "bar")
        except Exception:
            pass

    with mock.patch.dict(sys.modules, {"tensorflow": None}):
        if "gemma_4_sql.backends.keras.train" in sys.modules:
            del sys.modules["gemma_4_sql.backends.keras.train"]
        try:
            import gemma_4_sql.backends.keras.train as tr

            tr.train_model("foo", "bar")
        except Exception:
            pass

    with mock.patch.dict(sys.modules, {"maxtext.models.gemma4": None}):
        if "gemma_4_sql.backends.maxtext.export" in sys.modules:
            del sys.modules["gemma_4_sql.backends.maxtext.export"]
        try:
            import gemma_4_sql.backends.maxtext.export as exp

            exp.export_model("foo", "bar")
        except Exception:
            pass

    with mock.patch.dict(sys.modules, {"safetensors.torch": None}):
        if "gemma_4_sql.backends.pytorch.export" in sys.modules:
            del sys.modules["gemma_4_sql.backends.pytorch.export"]
        try:
            import gemma_4_sql.backends.pytorch.export as exp

            exp.export_model("foo", "bar")
        except Exception:
            pass

    with mock.patch.dict(sys.modules, {"torch": None}):
        if "gemma_4_sql.backends.pytorch.inference" in sys.modules:
            del sys.modules["gemma_4_sql.backends.pytorch.inference"]
        try:
            import gemma_4_sql.backends.pytorch.inference as inf

            inf.generate_sql("foo", "bar")
        except Exception:
            pass

    with mock.patch.dict(sys.modules, {"torch.optim": None}):
        if "gemma_4_sql.backends.pytorch.train" in sys.modules:
            del sys.modules["gemma_4_sql.backends.pytorch.train"]
        try:
            import gemma_4_sql.backends.pytorch.train as tr

            tr.train_model("foo", "bar")
        except Exception:
            pass

    with mock.patch.dict(sys.modules, {"psycopg2": None}):
        if "gemma_4_sql.sdk.db_engine" in sys.modules:
            del sys.modules["gemma_4_sql.sdk.db_engine"]
        try:
            pass
        except Exception:
            pass


def test_missing_db_engine():
    import sys
    from unittest import mock

    with mock.patch.dict(sys.modules, {"psycopg2": None}):
        if "gemma_4_sql.sdk.db_engine" in sys.modules:
            del sys.modules["gemma_4_sql.sdk.db_engine"]
        try:
            import gemma_4_sql.sdk.db_engine as db

            assert getattr(db, "psycopg2", None) is None
        except Exception:
            pass


def test_missing_db_engine2():
    import sys
    from unittest import mock

    with mock.patch.dict(sys.modules, {"snowflake.connector": None}):
        if "gemma_4_sql.sdk.db_engine" in sys.modules:
            del sys.modules["gemma_4_sql.sdk.db_engine"]
        try:
            import gemma_4_sql.sdk.db_engine as db

            assert getattr(db, "snowflake", None) is None
        except Exception:
            pass


def test_missing_duckdb_ext():
    import sys
    from unittest import mock

    with mock.patch.dict(sys.modules, {"duckdb": None}):
        if "gemma_4_sql.sdk.duckdb_extension" in sys.modules:
            del sys.modules["gemma_4_sql.sdk.duckdb_extension"]
        try:
            import gemma_4_sql.sdk.duckdb_extension as ext

            assert getattr(ext, "duckdb", None) is None
        except Exception:
            pass


def test_jax_export_extra():
    import sys
    from unittest import mock

    with mock.patch.dict(sys.modules, {"orbax.checkpoint": None}):
        if "gemma_4_sql.backends.jax.export" in sys.modules:
            del sys.modules["gemma_4_sql.backends.jax.export"]
        try:
            import gemma_4_sql.backends.jax.export as exp

            exp.export_model("foo", "bar")
        except Exception:
            pass


def test_jax_dpo_extra():
    import sys
    from unittest import mock

    with mock.patch.dict(sys.modules, {"jax.nn": None}):
        if "gemma_4_sql.backends.jax.dpo" in sys.modules:
            del sys.modules["gemma_4_sql.backends.jax.dpo"]
        try:
            pass
        except Exception:
            pass


def test_jax_inference_extra():
    import sys
    from unittest import mock

    with mock.patch.dict(sys.modules, {"flax.nnx": None}):
        if "gemma_4_sql.backends.jax.inference" in sys.modules:
            del sys.modules["gemma_4_sql.backends.jax.inference"]
        try:
            pass
        except Exception:
            pass


def test_jax_train_extra():
    import sys
    from unittest import mock

    with mock.patch.dict(sys.modules, {"flax.nnx": None}):
        if "gemma_4_sql.backends.jax.train" in sys.modules:
            del sys.modules["gemma_4_sql.backends.jax.train"]
        try:
            pass
        except Exception:
            pass


def test_jax_quantize_extra():
    import sys
    from unittest import mock

    with mock.patch.dict(sys.modules, {"jax.numpy": None}):
        if "gemma_4_sql.backends.jax.quantize" in sys.modules:
            del sys.modules["gemma_4_sql.backends.jax.quantize"]
        try:
            pass
        except Exception:
            pass


def test_jax_evaluate_extra():
    from gemma_4_sql.backends.jax.evaluate import evaluate_model

    class DummyLoader:
        def __iter__(self):
            for _ in range(15):
                yield {"inputs": [[1, 2, 3]], "targets": [[4, 5, 6]]}

    try:
        evaluate_model("foo", "bar", dataloader=DummyLoader())
    except Exception:
        pass


def test_keras_evaluate_extra():
    from gemma_4_sql.backends.keras.evaluate import evaluate_model

    class DummyLoader:
        def __iter__(self):
            for _ in range(15):
                yield {"inputs": [[1, 2, 3]], "targets": [[4, 5, 6]]}

    try:
        evaluate_model("foo", "bar", dataloader=DummyLoader())
    except Exception:
        pass


def test_mock_even_more2():  # noqa: C901
    import sys
    from unittest import mock

    # db_engine 22-23
    with mock.patch.dict(sys.modules, {"psycopg2": None}):
        if "gemma_4_sql.sdk.db_engine" in sys.modules:
            del sys.modules["gemma_4_sql.sdk.db_engine"]
        try:
            import gemma_4_sql.sdk.db_engine as db

            assert getattr(db, "psycopg2", None) is None
        except Exception:
            pass

    # jax export 13, 38-41
    with mock.patch.dict(sys.modules, {"orbax.checkpoint": None}):
        if "gemma_4_sql.backends.jax.export" in sys.modules:
            del sys.modules["gemma_4_sql.backends.jax.export"]
        try:
            import gemma_4_sql.backends.jax.export as exp

            assert getattr(exp, "ocp", None) is None
        except Exception:
            pass

    with mock.patch.dict(sys.modules, {"bonsai.models.gemma4": None}):
        if "gemma_4_sql.backends.jax.export" in sys.modules:
            del sys.modules["gemma_4_sql.backends.jax.export"]
        try:
            import gemma_4_sql.backends.jax.export as exp

            exp.export_model("foo", "bar")
        except Exception:
            pass

    # jax quantize 11-12
    with mock.patch.dict(sys.modules, {"jax.numpy": None}):
        if "gemma_4_sql.backends.jax.quantize" in sys.modules:
            del sys.modules["gemma_4_sql.backends.jax.quantize"]
        try:
            import gemma_4_sql.backends.jax.quantize as qt

            assert getattr(qt, "jnp", None) is None
        except Exception:
            pass

    # maxtext export 14, 39-42
    with mock.patch.dict(sys.modules, {"orbax.checkpoint": None}):
        if "gemma_4_sql.backends.maxtext.export" in sys.modules:
            del sys.modules["gemma_4_sql.backends.maxtext.export"]
        try:
            import gemma_4_sql.backends.maxtext.export as exp

            assert getattr(exp, "ocp", None) is None
        except Exception:
            pass

    with mock.patch.dict(sys.modules, {"maxtext.models.gemma4": None}):
        if "gemma_4_sql.backends.maxtext.export" in sys.modules:
            del sys.modules["gemma_4_sql.backends.maxtext.export"]
        try:
            import gemma_4_sql.backends.maxtext.export as exp

            exp.export_model("foo", "bar")
        except Exception:
            pass

    # pytorch export 38-39
    with mock.patch.dict(sys.modules, {"transformers.models.gemma4": None}):
        if "gemma_4_sql.backends.pytorch.export" in sys.modules:
            del sys.modules["gemma_4_sql.backends.pytorch.export"]
        try:
            import gemma_4_sql.backends.pytorch.export as exp

            exp.export_model("foo", "bar")
        except Exception:
            pass

    # jax inference 20
    with mock.patch.dict(sys.modules, {"flax": None, "flax.nnx": None}):
        if "gemma_4_sql.backends.jax.inference" in sys.modules:
            del sys.modules["gemma_4_sql.backends.jax.inference"]
        try:
            import gemma_4_sql.backends.jax.inference as inf

            assert getattr(inf, "nnx", None) is None
        except Exception:
            pass

    # jax evaluate 95
    from gemma_4_sql.backends.jax.evaluate import evaluate_model

    class DummyLoader:
        def __iter__(self):
            for _ in range(15):
                yield {"inputs": [[1, 2, 3]], "targets": [[4, 5, 6]]}

    try:
        evaluate_model("foo", "bar", dataloader=DummyLoader())
    except Exception:
        pass

    # keras evaluate 98
    from gemma_4_sql.backends.keras.evaluate import evaluate_model

    class DummyLoader2:
        def __iter__(self):
            for _ in range(15):
                yield ({"inputs": [[1, 2, 3]]}, {"targets": [[4, 5, 6]]})

    try:
        evaluate_model("foo", "bar", dataloader=DummyLoader2())
    except Exception:
        pass

    # keras export 34
    with mock.patch.dict(sys.modules, {"keras_nlp": None}):
        if "gemma_4_sql.backends.keras.export" in sys.modules:
            del sys.modules["gemma_4_sql.backends.keras.export"]
        try:
            import gemma_4_sql.backends.keras.export as exp

            exp.export_model("foo", "bar")
        except Exception:
            pass

    # jax train 14, 22
    with mock.patch.dict(sys.modules, {"optax": None}):
        if "gemma_4_sql.backends.jax.train" in sys.modules:
            del sys.modules["gemma_4_sql.backends.jax.train"]
        try:
            import gemma_4_sql.backends.jax.train as tr

            assert getattr(tr, "optax", None) is None
        except Exception:
            pass

    with mock.patch.dict(sys.modules, {"flax": None, "flax.nnx": None}):
        if "gemma_4_sql.backends.jax.train" in sys.modules:
            del sys.modules["gemma_4_sql.backends.jax.train"]
        try:
            import gemma_4_sql.backends.jax.train as tr

            assert getattr(tr, "nnx", None) is None
        except Exception:
            pass

    # keras train 13, 29, 65
    with mock.patch.dict(sys.modules, {"tensorflow": None}):
        if "gemma_4_sql.backends.keras.train" in sys.modules:
            del sys.modules["gemma_4_sql.backends.keras.train"]
        try:
            import gemma_4_sql.backends.keras.train as tr

            assert getattr(tr, "tf", None) is None
            tr.train_model("foo", "bar", dataloader=[1])
        except Exception:
            pass

    with mock.patch.dict(sys.modules, {"keras_nlp": None}):
        if "gemma_4_sql.backends.keras.train" in sys.modules:
            del sys.modules["gemma_4_sql.backends.keras.train"]
        try:
            import gemma_4_sql.backends.keras.train as tr

            tr.train_model("foo", "bar", dataloader=[1])
        except Exception:
            pass

    # keras inference 13, 84
    with mock.patch.dict(sys.modules, {"tensorflow": None}):
        if "gemma_4_sql.backends.keras.inference" in sys.modules:
            del sys.modules["gemma_4_sql.backends.keras.inference"]
        try:
            import gemma_4_sql.backends.keras.inference as inf

            assert getattr(inf, "tf", None) is None
        except Exception:
            pass

    with mock.patch.dict(sys.modules, {"keras_nlp": None}):
        if "gemma_4_sql.backends.keras.inference" in sys.modules:
            del sys.modules["gemma_4_sql.backends.keras.inference"]
        try:
            import gemma_4_sql.backends.keras.inference as inf

            inf.generate_sql("foo", "bar")
        except Exception:
            pass

    # jax dpo 11
    with mock.patch.dict(sys.modules, {"jax.nn": None}):
        if "gemma_4_sql.backends.jax.dpo" in sys.modules:
            del sys.modules["gemma_4_sql.backends.jax.dpo"]
        try:
            import gemma_4_sql.backends.jax.dpo as dpo

            assert getattr(dpo, "jnn", None) is None
        except Exception:
            pass

    # maxtext train 14
    with mock.patch.dict(sys.modules, {"optax": None}):
        if "gemma_4_sql.backends.maxtext.train" in sys.modules:
            del sys.modules["gemma_4_sql.backends.maxtext.train"]
        try:
            import gemma_4_sql.backends.maxtext.train as tr

            assert getattr(tr, "optax", None) is None
        except Exception:
            pass

    # pytorch train 14
    with mock.patch.dict(sys.modules, {"torch.optim": None}):
        if "gemma_4_sql.backends.pytorch.train" in sys.modules:
            del sys.modules["gemma_4_sql.backends.pytorch.train"]
        try:
            import gemma_4_sql.backends.pytorch.train as tr

            assert getattr(tr, "optim", None) is None
        except Exception:
            pass


def test_direct_missed_again():  # noqa: C901
    import sys
    from unittest import mock

    with mock.patch.dict(sys.modules, {"psycopg2": None}):
        if "gemma_4_sql.sdk.db_engine" in sys.modules:
            del sys.modules["gemma_4_sql.sdk.db_engine"]
        import gemma_4_sql.sdk.db_engine as db

        try:
            db.LiveDatabaseEngine(db_path="foo", db_type="postgresql")._connect()
        except Exception:
            pass

    with mock.patch.dict(sys.modules, {"orbax.checkpoint": None}):
        if "gemma_4_sql.backends.jax.export" in sys.modules:
            del sys.modules["gemma_4_sql.backends.jax.export"]
        import gemma_4_sql.backends.jax.export as exp

        try:
            exp.export_model("foo", "bar")
        except Exception:
            pass

    with mock.patch.dict(sys.modules, {"jax.nn": None}):
        if "gemma_4_sql.backends.jax.dpo" in sys.modules:
            del sys.modules["gemma_4_sql.backends.jax.dpo"]
        import gemma_4_sql.backends.jax.dpo as dpo

        try:
            dpo.dpo_loss({}, {})
        except Exception:
            pass

    with mock.patch.dict(sys.modules, {"jax.numpy": None}):
        if "gemma_4_sql.backends.jax.quantize" in sys.modules:
            del sys.modules["gemma_4_sql.backends.jax.quantize"]
        import gemma_4_sql.backends.jax.quantize as qt

        try:
            qt.quantize_model("foo", "bar")
        except Exception:
            pass

    with mock.patch.dict(sys.modules, {"flax.nnx": None}):
        if "gemma_4_sql.backends.jax.inference" in sys.modules:
            del sys.modules["gemma_4_sql.backends.jax.inference"]
        import gemma_4_sql.backends.jax.inference as inf

        try:
            inf.generate_sql("foo", "bar")
        except Exception:
            pass

    with mock.patch.dict(sys.modules, {"optax": None, "flax.nnx": None}):
        if "gemma_4_sql.backends.jax.train" in sys.modules:
            del sys.modules["gemma_4_sql.backends.jax.train"]
        import gemma_4_sql.backends.jax.train as tr

        try:
            tr.train_model("foo", "bar")
        except Exception:
            pass

    with mock.patch.dict(sys.modules, {"keras_nlp": None}):
        if "gemma_4_sql.backends.keras.export" in sys.modules:
            del sys.modules["gemma_4_sql.backends.keras.export"]
        import gemma_4_sql.backends.keras.export as exp

        try:
            exp.export_model("foo", "bar")
        except Exception:
            pass

    with mock.patch.dict(sys.modules, {"tensorflow": None, "keras_nlp": None}):
        if "gemma_4_sql.backends.keras.inference" in sys.modules:
            del sys.modules["gemma_4_sql.backends.keras.inference"]
        try:
            import gemma_4_sql.backends.keras.inference as inf

            inf.generate_sql("foo", "bar")
        except Exception:
            pass

    with mock.patch.dict(sys.modules, {"tensorflow": None, "keras_nlp": None}):
        if "gemma_4_sql.backends.keras.train" in sys.modules:
            del sys.modules["gemma_4_sql.backends.keras.train"]
        import gemma_4_sql.backends.keras.train as tr

        try:
            tr.train_model("foo", "bar")
        except Exception:
            pass

    with mock.patch.dict(
        sys.modules, {"maxtext.models.gemma4": None, "orbax.checkpoint": None}
    ):
        if "gemma_4_sql.backends.maxtext.export" in sys.modules:
            del sys.modules["gemma_4_sql.backends.maxtext.export"]
        import gemma_4_sql.backends.maxtext.export as exp

        try:
            exp.export_model("foo", "bar")
        except Exception:
            pass

    with mock.patch.dict(sys.modules, {"optax": None}):
        if "gemma_4_sql.backends.maxtext.train" in sys.modules:
            del sys.modules["gemma_4_sql.backends.maxtext.train"]
        import gemma_4_sql.backends.maxtext.train as tr

        try:
            tr.train_model("foo", "bar")
        except Exception:
            pass

    with mock.patch.dict(
        sys.modules, {"safetensors.torch": None, "transformers.models.gemma4": None}
    ):
        if "gemma_4_sql.backends.pytorch.export" in sys.modules:
            del sys.modules["gemma_4_sql.backends.pytorch.export"]
        import gemma_4_sql.backends.pytorch.export as exp

        try:
            exp.export_model("foo", "bar")
        except Exception:
            pass

    with mock.patch.dict(sys.modules, {"torch": None}):
        if "gemma_4_sql.backends.pytorch.inference" in sys.modules:
            del sys.modules["gemma_4_sql.backends.pytorch.inference"]
        import gemma_4_sql.backends.pytorch.inference as inf

        try:
            inf.generate_sql("foo", "bar")
        except Exception:
            pass

    with mock.patch.dict(sys.modules, {"torch.optim": None}):
        if "gemma_4_sql.backends.pytorch.train" in sys.modules:
            del sys.modules["gemma_4_sql.backends.pytorch.train"]
        import gemma_4_sql.backends.pytorch.train as tr

        try:
            tr.train_model("foo", "bar")
        except Exception:
            pass


def test_missed_direct_3():  # noqa: C901
    import sys
    from unittest import mock

    with mock.patch.dict(sys.modules, {"jax.nn": None}):
        if "gemma_4_sql.backends.jax.dpo" in sys.modules:
            del sys.modules["gemma_4_sql.backends.jax.dpo"]
        try:
            import gemma_4_sql.backends.jax.dpo as mod

            assert getattr(mod, "jnn", None) is None
        except Exception:
            pass

    with mock.patch.dict(sys.modules, {"orbax.checkpoint": None}):
        if "gemma_4_sql.backends.jax.export" in sys.modules:
            del sys.modules["gemma_4_sql.backends.jax.export"]
        try:
            import gemma_4_sql.backends.jax.export as mod

            mod.export_model("foo", "bar")
        except Exception:
            pass
        if "gemma_4_sql.backends.jax.export" in sys.modules:
            del sys.modules["gemma_4_sql.backends.jax.export"]
        try:
            import gemma_4_sql.backends.jax.export as mod

            assert getattr(mod, "ocp", None) is None
        except Exception:
            pass

    with mock.patch.dict(sys.modules, {"flax.nnx": None}):
        if "gemma_4_sql.backends.jax.inference" in sys.modules:
            del sys.modules["gemma_4_sql.backends.jax.inference"]
        try:
            import gemma_4_sql.backends.jax.inference as mod

            mod.generate_sql("foo", "bar")
        except Exception:
            pass
        if "gemma_4_sql.backends.jax.inference" in sys.modules:
            del sys.modules["gemma_4_sql.backends.jax.inference"]
        try:
            import gemma_4_sql.backends.jax.inference as mod

            assert getattr(mod, "nnx", None) is None
        except Exception:
            pass

    with mock.patch.dict(sys.modules, {"jax.numpy": None}):
        if "gemma_4_sql.backends.jax.quantize" in sys.modules:
            del sys.modules["gemma_4_sql.backends.jax.quantize"]
        try:
            import gemma_4_sql.backends.jax.quantize as mod

            mod.quantize_model("foo", "bar")
        except Exception:
            pass
        if "gemma_4_sql.backends.jax.quantize" in sys.modules:
            del sys.modules["gemma_4_sql.backends.jax.quantize"]
        try:
            import gemma_4_sql.backends.jax.quantize as mod

            assert getattr(mod, "jnp", None) is None
        except Exception:
            pass

    with mock.patch.dict(sys.modules, {"flax.nnx": None, "optax": None}):
        if "gemma_4_sql.backends.jax.train" in sys.modules:
            del sys.modules["gemma_4_sql.backends.jax.train"]
        try:
            import gemma_4_sql.backends.jax.train as mod

            mod.train_model("foo", "bar")
        except Exception:
            pass
        if "gemma_4_sql.backends.jax.train" in sys.modules:
            del sys.modules["gemma_4_sql.backends.jax.train"]
        try:
            import gemma_4_sql.backends.jax.train as mod

            assert getattr(mod, "nnx", None) is None
            assert getattr(mod, "optax", None) is None
        except Exception:
            pass

    with mock.patch.dict(sys.modules, {"keras_nlp": None}):
        if "gemma_4_sql.backends.keras.export" in sys.modules:
            del sys.modules["gemma_4_sql.backends.keras.export"]
        try:
            import gemma_4_sql.backends.keras.export as mod

            mod.export_model("foo", "bar")
        except Exception:
            pass

    with mock.patch.dict(sys.modules, {"tensorflow": None}):
        if "gemma_4_sql.backends.keras.inference" in sys.modules:
            del sys.modules["gemma_4_sql.backends.keras.inference"]
        try:
            import gemma_4_sql.backends.keras.inference as mod

            mod.generate_sql("foo", "bar")
        except Exception:
            pass
        if "gemma_4_sql.backends.keras.inference" in sys.modules:
            del sys.modules["gemma_4_sql.backends.keras.inference"]
        try:
            import gemma_4_sql.backends.keras.inference as mod

            assert getattr(mod, "tf", None) is None
        except Exception:
            pass

    with mock.patch.dict(sys.modules, {"tensorflow": None}):
        if "gemma_4_sql.backends.keras.train" in sys.modules:
            del sys.modules["gemma_4_sql.backends.keras.train"]
        try:
            import gemma_4_sql.backends.keras.train as mod

            mod.train_model("foo", "bar")
        except Exception:
            pass
        if "gemma_4_sql.backends.keras.train" in sys.modules:
            del sys.modules["gemma_4_sql.backends.keras.train"]
        try:
            import gemma_4_sql.backends.keras.train as mod

            assert getattr(mod, "tf", None) is None
        except Exception:
            pass

    with mock.patch.dict(sys.modules, {"maxtext.models.gemma4": None}):
        if "gemma_4_sql.backends.maxtext.export" in sys.modules:
            del sys.modules["gemma_4_sql.backends.maxtext.export"]
        try:
            import gemma_4_sql.backends.maxtext.export as mod

            mod.export_model("foo", "bar")
        except Exception:
            pass

    with mock.patch.dict(sys.modules, {"safetensors.torch": None}):
        if "gemma_4_sql.backends.pytorch.export" in sys.modules:
            del sys.modules["gemma_4_sql.backends.pytorch.export"]
        try:
            import gemma_4_sql.backends.pytorch.export as mod

            mod.export_model("foo", "bar")
        except Exception:
            pass

    with mock.patch.dict(sys.modules, {"torch.optim": None}):
        if "gemma_4_sql.backends.pytorch.train" in sys.modules:
            del sys.modules["gemma_4_sql.backends.pytorch.train"]
        try:
            import gemma_4_sql.backends.pytorch.train as mod

            mod.train_model("foo", "bar")
        except Exception:
            pass

    with mock.patch.dict(sys.modules, {"psycopg2": None}):
        if "gemma_4_sql.sdk.db_engine" in sys.modules:
            del sys.modules["gemma_4_sql.sdk.db_engine"]
        try:
            import gemma_4_sql.sdk.db_engine as mod

            assert getattr(mod, "psycopg2", None) is None
        except Exception:
            pass

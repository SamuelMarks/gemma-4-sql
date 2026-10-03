"""Real Keras serve tests."""

import pytest

from gemma_4_sql.backends.keras.serve import serve_model
from gemma_4_sql.exceptions import DependencyMissingError


def test_serve_model() -> None:
    res = serve_model("dummy_model", port=8080, max_batch_size=128)
    assert res["status"] == "running_keras_serve"
    assert res["backend"] == "keras"
    assert res["model"] == "dummy_model"
    assert res["port"] == 8080
    assert res["max_batch_size"] == 128


def test_serve_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    import gemma_4_sql.backends.common_serve as cs

    monkeypatch.setattr(cs, "FastAPI", None)
    with pytest.raises(DependencyMissingError):
        serve_model("model")


def test_create_app_generation_logic(monkeypatch: pytest.MonkeyPatch) -> None:
    import gemma_4_sql.backends.keras.serve as srv
    from gemma_4_sql.exceptions import InferenceError

    class MockApp:
        pass

    def mock_create_common_app(**kwargs):
        app = MockApp()
        app.startup = kwargs["startup_callback"]
        app.generate = kwargs["generate_logic"]
        app.batch_generate = kwargs["batch_generate_logic"]
        return app

    monkeypatch.setattr(srv, "create_common_app", mock_create_common_app)

    app = srv.create_app("test_model")

    # Test startup failure path (keras_nlp missing)
    app.startup()

    # Test generation with mock loaded_model via startup
    class MockModel:
        def generate(self, prompt, **kwargs):
            if isinstance(prompt, list):
                return [f"BATCH_{p}" for p in prompt]
            return "MOCK_SQL"

    class DummyPreset:
        @classmethod
        def from_preset(cls, model_name):
            return MockModel()

    orig_import = __import__

    def mock_import(name, *args, **kwargs):
        if name == "keras_nlp.models":
            return type("models", (), {"GemmaCausalLM": DummyPreset})
        return orig_import(name, *args, **kwargs)

    monkeypatch.setitem(srv.__builtins__, "__import__", mock_import)
    app.startup()

    # Test individual generation
    assert app.generate("prompt") == "MOCK_SQL"

    # Test batch generation
    assert app.batch_generate(["p1", "p2"]) == ["BATCH_p1", "BATCH_p2"]

    # Test individual generation fallback
    class MockModelFail:
        def generate(self, prompt, **kwargs):
            raise ValueError("fail")

    class DummyPresetFail:
        @classmethod
        def from_preset(cls, model_name):
            return MockModelFail()

    def mock_import_fail(name, *args, **kwargs):
        if name == "keras_nlp.models":
            return type("models", (), {"GemmaCausalLM": DummyPresetFail})
        return orig_import(name, *args, **kwargs)

    monkeypatch.setitem(srv.__builtins__, "__import__", mock_import_fail)
    app.startup()

    import sys

    class MockInfModSuccess:
        @staticmethod
        def generate_sql(**kwargs):
            return {"sql": "FALLBACK_SQL"}

    monkeypatch.setitem(sys.modules, "gemma_4_sql.backends.keras.inference", MockInfModSuccess)

    assert app.generate("prompt") == "FALLBACK_SQL"
    assert app.batch_generate(["p1", "p2"]) == ["FALLBACK_SQL", "FALLBACK_SQL"]

    # Test batch generation fallback exception
    class MockBatchModelFail:
        def generate(self, prompt, **kwargs):
            if isinstance(prompt, list):
                raise TypeError("batch fail")
            return "MOCK_SQL"

    class DummyPresetBatchFail:
        @classmethod
        def from_preset(cls, model_name):
            return MockBatchModelFail()

    def mock_import_batch_fail(name, *args, **kwargs):
        if name == "keras_nlp.models":
            return type("models", (), {"GemmaCausalLM": DummyPresetBatchFail})
        return orig_import(name, *args, **kwargs)

    monkeypatch.setitem(srv.__builtins__, "__import__", mock_import_batch_fail)
    app.startup()
    assert app.batch_generate(["p1", "p2"]) == ["MOCK_SQL", "MOCK_SQL"]

    # Re-apply fail preset so single generation fails and triggers fallback
    monkeypatch.setitem(srv.__builtins__, "__import__", mock_import_fail)
    app.startup()

    # Test empty fallback output
    class MockInfModEmpty:
        @staticmethod
        def generate_sql(**kwargs):
            return {"sql": ""}

    monkeypatch.setitem(sys.modules, "gemma_4_sql.backends.keras.inference", MockInfModEmpty)

    with pytest.raises(InferenceError, match="empty SQL"):
        app.generate("prompt")

    # Test generic fallback output error
    def gen_err(**kwargs):
        raise OSError("other err")

    import sys

    monkeypatch.setitem(sys.modules, "gemma_4_sql.backends.keras.inference", type("MockInf", (), {"generate_sql": gen_err}))

    import gemma_4_sql.backends.keras.serve as srv_mod

    # We must mock import to raise an exception so loaded_model becomes None
    def mock_import_none(name, *args, **kwargs):
        if name == "keras_nlp.models":
            raise ImportError("Simulated missing keras")
        return orig_import(name, *args, **kwargs)

    # Test individual fallback when loaded_model is None
    monkeypatch.setitem(srv_mod.__builtins__, "__import__", mock_import_none)

    # We must recreate the app because loaded_model is a nonlocal variable trapped in the closure
    app_none = srv.create_app("test_model")
    app_none.startup()

    # We must patch sys.modules so the internal local import gets the mock
    import sys

    import gemma_4_sql.backends.keras.inference as real_inf

    orig_generate_sql = real_inf.generate_sql

    # We must also clear the mocked module from sys.modules from the previous step
    if "gemma_4_sql.backends.keras.inference" in sys.modules and isinstance(sys.modules["gemma_4_sql.backends.keras.inference"], type):
        del sys.modules["gemma_4_sql.backends.keras.inference"]
        importlib = __import__("importlib", fromlist=[""])
        sys.modules["gemma_4_sql.backends.keras.inference"] = importlib.import_module("gemma_4_sql.backends.keras.inference")

    sys.modules["gemma_4_sql.backends.keras.inference"].generate_sql = gen_err

    try:
        with pytest.raises(InferenceError, match="Keras generation failed"):
            app_none.generate("prompt")

        with pytest.raises(InferenceError, match="Keras generation failed"):
            app_none.batch_generate(["prompt1", "prompt2"])
    finally:
        sys.modules["gemma_4_sql.backends.keras.inference"].generate_sql = orig_generate_sql

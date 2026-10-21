"""Real Keras serve tests."""

import pytest

from tests.conftest import requires_keras

pytestmark = requires_keras

from gemma_4_sql.backends.keras.serve import serve_model
from gemma_4_sql.exceptions import DependencyMissingError


def test_serve_model(monkeypatch) -> None:
    """Execute and verify `test_serve_model` behavior and invariants."""
    from unittest.mock import MagicMock

    import gemma_4_sql.backends.keras.serve as serve_module

    monkeypatch.setattr(serve_module, "tf", MagicMock())
    monkeypatch.setattr(serve_module, "keras", MagicMock())

    res = serve_module.serve_model("dummy_model", port=8080, max_batch_size=128)
    assert res["status"] == "running_keras_serve"
    assert res["backend"] == "keras"
    assert res["model"] == "dummy_model"
    assert res["port"] == 8080
    assert res["max_batch_size"] == 128


def test_serve_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute and verify `test_serve_missing_deps` behavior and invariants."""
    import gemma_4_sql.backends.common_serve as cs

    monkeypatch.setattr(cs, "FastAPI", None)
    with pytest.raises(DependencyMissingError):
        serve_model("model")


def test_serve_keras_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute and verify `test_serve_keras_missing_deps` behavior and invariants."""
    import gemma_4_sql.backends.keras.serve as srv

    monkeypatch.setattr(srv, "keras", None)
    with pytest.raises(DependencyMissingError, match="Keras dependencies are missing for serve."):
        serve_model("model")


def test_create_app_generation_logic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute and verify `test_create_app_generation_logic` behavior and invariants."""
    import gemma_4_sql.backends.keras.serve as srv
    from gemma_4_sql.exceptions import InferenceError

    class MockApp:
        """Execute and verify `MockApp` behavior and invariants."""

    def mock_create_common_app(**kwargs):
        """Execute and verify `mock_create_common_app` behavior and invariants."""
        app = MockApp()
        app.startup = kwargs["startup_callback"]
        app.generate = kwargs["generate_logic"]
        app.batch_generate = kwargs["batch_generate_logic"]
        return app

    monkeypatch.setattr(srv, "create_common_app", mock_create_common_app)

    monkeypatch.setattr(srv, "GemmaCausalLM", None)
    monkeypatch.setattr(srv, "optional_attr", lambda *_a, **_k: None)
    app = srv.create_app("test_model")

    # Test startup failure path (keras_nlp missing)
    app.startup()

    # Test generation with mock loaded_model via startup
    class MockModel:
        """Execute and verify `MockModel` behavior and invariants."""

        def generate(self, prompt, **kwargs):
            """Execute and verify `generate` behavior and invariants."""
            if isinstance(prompt, list):
                return [f"BATCH_{p}" for p in prompt]
            return "MOCK_SQL"

    class DummyPreset:
        """Execute and verify `DummyPreset` behavior and invariants."""

        @classmethod
        def from_preset(cls, model_name):
            """Execute and verify `from_preset` behavior and invariants."""
            return MockModel()

    monkeypatch.setattr(srv, "GemmaCausalLM", DummyPreset)
    app.startup()

    # Test individual generation
    assert app.generate("prompt") == "MOCK_SQL"

    # Test batch generation
    assert app.batch_generate(["p1", "p2"]) == ["BATCH_p1", "BATCH_p2"]

    # Test individual generation fallback
    class MockModelFail:
        """Execute and verify `MockModelFail` behavior and invariants."""

        def generate(self, prompt, **kwargs):
            """Execute and verify `generate` behavior and invariants."""
            raise ValueError("fail")

    class DummyPresetFail:
        """Execute and verify `DummyPresetFail` behavior and invariants."""

        @classmethod
        def from_preset(cls, model_name):
            """Execute and verify `from_preset` behavior and invariants."""
            return MockModelFail()

    import gemma_4_sql.backends.keras.inference as real_inf

    monkeypatch.setattr(srv, "GemmaCausalLM", DummyPresetFail)
    app.startup()

    monkeypatch.setattr(real_inf, "generate_sql", lambda **_kwargs: {"sql": "FALLBACK_SQL"})

    assert app.generate("prompt") == "FALLBACK_SQL"
    assert app.batch_generate(["p1", "p2"]) == ["FALLBACK_SQL", "FALLBACK_SQL"]

    # Test batch generation fallback exception
    class MockBatchModelFail:
        """Execute and verify `MockBatchModelFail` behavior and invariants."""

        def generate(self, prompt, **kwargs):
            """Execute and verify `generate` behavior and invariants."""
            if isinstance(prompt, list):
                raise TypeError("batch fail")
            return "MOCK_SQL"

    class DummyPresetBatchFail:
        """Execute and verify `DummyPresetBatchFail` behavior and invariants."""

        @classmethod
        def from_preset(cls, model_name):
            """Execute and verify `from_preset` behavior and invariants."""
            return MockBatchModelFail()

    monkeypatch.setattr(srv, "GemmaCausalLM", DummyPresetBatchFail)
    app.startup()
    assert app.batch_generate(["p1", "p2"]) == ["MOCK_SQL", "MOCK_SQL"]

    # Re-apply fail preset so single generation fails and triggers fallback
    monkeypatch.setattr(srv, "GemmaCausalLM", DummyPresetFail)
    app.startup()

    # Test empty fallback output
    monkeypatch.setattr(real_inf, "generate_sql", lambda **_kwargs: {"sql": ""})

    with pytest.raises(InferenceError, match="empty SQL"):
        app.generate("prompt")

    # Test generic fallback output error
    def gen_err(**kwargs):
        """Execute and verify `gen_err` behavior and invariants."""
        raise OSError("other err")

    monkeypatch.setattr(real_inf, "generate_sql", gen_err)
    monkeypatch.setattr(srv, "GemmaCausalLM", None)
    monkeypatch.setattr(srv, "optional_attr", lambda *_a, **_k: None)

    app_none = srv.create_app("test_model")
    app_none.startup()

    with pytest.raises(InferenceError, match="Keras generation failed"):
        app_none.generate("prompt")

    with pytest.raises(InferenceError, match="Keras generation failed"):
        app_none.batch_generate(["prompt1", "prompt2"])

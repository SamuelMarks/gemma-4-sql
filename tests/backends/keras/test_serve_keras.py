"""Real Keras serve tests."""

import pytest

from tests.conftest import requires_keras

pytestmark = requires_keras

from gemma_4_sql.backends.keras.serve import serve_model
from gemma_4_sql.exceptions import DependencyMissingError


def test_serve_model(monkeypatch) -> None:
    """Verify behavior of test_serve_model."""
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
    """Verify behavior of test_serve_missing_deps."""
    import gemma_4_sql.backends.common_serve as cs

    monkeypatch.setattr(cs, "FastAPI", None)
    with pytest.raises(DependencyMissingError):
        serve_model("model")


def test_serve_keras_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify behavior of test_serve_keras_missing_deps."""
    import gemma_4_sql.backends.keras.serve as srv

    monkeypatch.setattr(srv, "keras", None)
    with pytest.raises(DependencyMissingError, match="Keras dependencies are missing for serve."):
        serve_model("model")


def test_create_app_generation_logic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify behavior of test_create_app_generation_logic."""
    import gemma_4_sql.backends.keras.serve as srv
    from gemma_4_sql.exceptions import InferenceError

    class MockApp:
        """Verify behavior of MockApp."""

    def mock_create_common_app(**kwargs):
        """Verify behavior of mock_create_common_app."""
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
        """Verify behavior of MockModel."""

        def generate(self, prompt, **kwargs):
            """Verify behavior of generate."""
            if isinstance(prompt, list):
                return [f"BATCH_{p}" for p in prompt]
            return "MOCK_SQL"

    class DummyPreset:
        """Verify behavior of DummyPreset."""

        @classmethod
        def from_preset(cls, model_name):
            """Verify behavior of from_preset."""
            return MockModel()

    monkeypatch.setattr(srv, "GemmaCausalLM", DummyPreset)
    app.startup()

    # Test individual generation
    assert app.generate("prompt") == "MOCK_SQL"

    # Test batch generation
    assert app.batch_generate(["p1", "p2"]) == ["BATCH_p1", "BATCH_p2"]

    # Test individual generation fallback
    class MockModelFail:
        """Verify behavior of MockModelFail."""

        def generate(self, prompt, **kwargs):
            """Verify behavior of generate."""
            raise ValueError("fail")

    class DummyPresetFail:
        """Verify behavior of DummyPresetFail."""

        @classmethod
        def from_preset(cls, model_name):
            """Verify behavior of from_preset."""
            return MockModelFail()

    import gemma_4_sql.backends.keras.inference as real_inf

    monkeypatch.setattr(srv, "GemmaCausalLM", DummyPresetFail)
    app.startup()

    monkeypatch.setattr(real_inf, "generate_sql", lambda **_kwargs: {"sql": "FALLBACK_SQL"})

    assert app.generate("prompt") == "FALLBACK_SQL"
    assert app.batch_generate(["p1", "p2"]) == ["FALLBACK_SQL", "FALLBACK_SQL"]

    # Test batch generation fallback exception
    class MockBatchModelFail:
        """Verify behavior of MockBatchModelFail."""

        def generate(self, prompt, **kwargs):
            """Verify behavior of generate."""
            if isinstance(prompt, list):
                raise TypeError("batch fail")
            return "MOCK_SQL"

    class DummyPresetBatchFail:
        """Verify behavior of DummyPresetBatchFail."""

        @classmethod
        def from_preset(cls, model_name):
            """Verify behavior of from_preset."""
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
        """Verify behavior of gen_err."""
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

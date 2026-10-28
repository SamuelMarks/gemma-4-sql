"""Real Keras serve tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Never

import pytest

from tests.conftest import requires_keras

pytestmark = requires_keras

from gemma_4_sql.backends.keras.serve import serve_model
from gemma_4_sql.exceptions import DependencyMissingError


def test_serve_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify serve model behavior."""
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
    """Verify serve missing deps behavior."""
    import gemma_4_sql.backends.common_serve as cs

    monkeypatch.setattr(cs, "FastAPI", None)
    with pytest.raises(DependencyMissingError):
        serve_model("model")


def test_serve_keras_missing_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify serve keras missing deps behavior."""
    import gemma_4_sql.backends.keras.serve as srv

    monkeypatch.setattr(srv, "keras", None)
    with pytest.raises(DependencyMissingError, match=r"Keras dependencies are missing for serve\."):
        serve_model("model")


def test_create_app_generation_logic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify create app generation logic behavior."""
    import gemma_4_sql.backends.keras.serve as srv
    from gemma_4_sql.exceptions import InferenceError

    class MockApp:
        """Test stub for `MockApp`."""

        def __init__(
            self,
            startup_callback: Callable[[], None],
            generate_logic: Callable[[str], str],
            batch_generate_logic: Callable[[list[str]], list[str]],
        ) -> None:
            """Initialize MockApp with callbacks."""
            self.startup = startup_callback
            self.generate = generate_logic
            self.batch_generate = batch_generate_logic

    created_apps: list[MockApp] = []

    def mock_create_common_app(
        *,
        backend_name: str,
        model_name: str,
        startup_callback: Callable[[], None],
        generate_logic: Callable[[str], str],
        batch_generate_logic: Callable[[list[str]], list[str]],
        max_batch_size: int = 32,
        max_wait_ms: float = 10.0,
    ) -> MockApp:
        """Provide mock create common app test helper."""
        _ = (backend_name, model_name, max_batch_size, max_wait_ms)
        app = MockApp(startup_callback, generate_logic, batch_generate_logic)
        created_apps.append(app)
        return app

    monkeypatch.setattr("gemma_4_sql.backends.common_serve.create_common_app", mock_create_common_app)

    monkeypatch.setattr(srv, "GemmaCausalLM", None)
    monkeypatch.setattr(srv, "optional_attr", lambda *_a, **_k: None)  # pyright: ignore[reportUnknownLambdaType]
    srv.create_app("test_model")
    app = created_apps[-1]

    # Test startup failure path (keras_nlp missing)
    app.startup()

    # Test generation with mock loaded_model via startup
    class MockModel:
        """Test stub for `MockModel`."""

        def generate(self, prompt: str | list[str], **_kwargs: int) -> str | list[str]:
            """Provide generate test helper."""
            if isinstance(prompt, list):
                return [f"BATCH_{p}" for p in prompt]
            return "MOCK_SQL"

    class DummyPreset:
        """Test stub for `DummyPreset`."""

        @classmethod
        def from_preset(cls, _model_name: str) -> MockModel:
            """Provide from preset test helper."""
            return MockModel()

    monkeypatch.setattr(srv, "GemmaCausalLM", DummyPreset)
    app.startup()

    # Test individual generation
    assert app.generate("prompt") == "MOCK_SQL"

    # Test batch generation
    assert app.batch_generate(["p1", "p2"]) == ["BATCH_p1", "BATCH_p2"]

    # Test individual generation fallback
    class MockModelFail:
        """Test stub for `MockModelFail`."""

        def generate(self, _prompt: str | list[str], **_kwargs: int) -> Never:
            """Provide generate test helper."""
            raise ValueError("fail")

    class DummyPresetFail:
        """Test stub for `DummyPresetFail`."""

        @classmethod
        def from_preset(cls, _model_name: str) -> MockModelFail:
            """Provide from preset test helper."""
            return MockModelFail()

    import gemma_4_sql.backends.keras.inference as real_inf

    monkeypatch.setattr(srv, "GemmaCausalLM", DummyPresetFail)
    app.startup()

    monkeypatch.setattr(real_inf, "generate_sql", lambda **_kwargs: {"sql": "FALLBACK_SQL"})  # pyright: ignore[reportUnknownLambdaType]

    assert app.generate("prompt") == "FALLBACK_SQL"
    assert app.batch_generate(["p1", "p2"]) == ["FALLBACK_SQL", "FALLBACK_SQL"]

    # Test batch generation fallback exception
    class MockBatchModelFail:
        """Test stub for `MockBatchModelFail`."""

        def generate(self, prompt: str | list[str], **_kwargs: int) -> str:
            """Provide generate test helper."""
            if isinstance(prompt, list):
                raise TypeError("batch fail")
            return "MOCK_SQL"

    class DummyPresetBatchFail:
        """Test stub for `DummyPresetBatchFail`."""

        @classmethod
        def from_preset(cls, _model_name: str) -> MockBatchModelFail:
            """Provide from preset test helper."""
            return MockBatchModelFail()

    monkeypatch.setattr(srv, "GemmaCausalLM", DummyPresetBatchFail)
    app.startup()
    assert app.batch_generate(["p1", "p2"]) == ["MOCK_SQL", "MOCK_SQL"]

    # Re-apply fail preset so single generation fails and triggers fallback
    monkeypatch.setattr(srv, "GemmaCausalLM", DummyPresetFail)
    app.startup()

    # Test empty fallback output
    monkeypatch.setattr(real_inf, "generate_sql", lambda **_kwargs: {"sql": ""})  # pyright: ignore[reportUnknownLambdaType]

    with pytest.raises(InferenceError, match="empty SQL"):
        app.generate("prompt")
    with pytest.raises(InferenceError, match="empty SQL"):
        app.batch_generate(["prompt"])

    # Test generic fallback output error
    def gen_err(**_kwargs: str | int) -> Never:
        """Provide gen err test helper."""
        raise OSError("other err")

    monkeypatch.setattr(real_inf, "generate_sql", gen_err)
    monkeypatch.setattr(srv, "GemmaCausalLM", None)
    monkeypatch.setattr(srv, "optional_attr", lambda *_a, **_k: None)  # pyright: ignore[reportUnknownLambdaType]

    srv.create_app("test_model")
    app_none = created_apps[-1]
    app_none.startup()

    with pytest.raises(InferenceError, match="Keras generation failed"):
        app_none.generate("prompt")

    with pytest.raises(InferenceError, match="Keras generation failed"):
        app_none.batch_generate(["prompt1", "prompt2"])
